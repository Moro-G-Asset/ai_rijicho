from __future__ import annotations

import os
from typing import List, Dict, Any

import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==== LINE SDK ====
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhook import WebhookParser
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent

# ========= 設定 =========

CSV_PATH = os.getenv("KIYAKU_CSV_PATH", "kiyaku+saisoku_A02.csv")

# OpenAI（無くても動く）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None  # ライブラリ未インストール時の保険

# LINE 用設定（環境変数から読む想定）
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")

if not LINE_CHANNEL_SECRET or not LINE_CHANNEL_ACCESS_TOKEN:
    # ここで落とさず、ログを残すだけにしておく
    print("⚠ LINE_CHANNEL_SECRET / LINE_CHANNEL_ACCESS_TOKEN が設定されていません。LINE連携は動作しません。")

line_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
line_parser = WebhookParser(LINE_CHANNEL_SECRET)


# ========= データ読み込み & 前処理 =========

def load_articles(csv_path: str) -> List[Dict[str, Any]]:
    """
    kiyaku+saisoku_A02.csv を読み込み、
    kind / id / heading ごとに 1条単位で集約したリストを返す。
    """
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    # 1条ごとにまとめる（text を結合、keywords はユニーク結合）
    grouped = (
        df.groupby(["kind", "id", "heading"], as_index=False)
        .agg(
            {
                "text": lambda s: "\n".join([t for t in s if t]),
                "keywords": lambda s: ", ".join(sorted({k for k in s if k})),
            }
        )
    )

    records: List[Dict[str, Any]] = []
    for _, row in grouped.iterrows():
        kind = str(row["kind"])
        art_id = str(row["id"])
        heading = str(row["heading"])
        text = str(row["text"])
        keywords = str(row["keywords"])

        label = make_label(kind, art_id)
        search_text = " ".join(
            [
                kind,
                art_id,
                heading,
                keywords,
                text,
            ]
        )

        records.append(
            {
                "kind": kind,
                "id": art_id,
                "heading": heading,
                "text": text,
                "keywords": keywords,
                "label": label,
                "search_text": search_text,
            }
        )

    return records


def make_label(kind: str, art_id: str) -> str:
    """
    「管理規約第○条」「ペット飼育細則第3条」のような正式ラベルを生成する。
    """
    if kind == "規約":
        base = "管理規約"
    else:
        base = kind

    art_id_str = str(art_id)
    try:
        n = int(float(art_id_str))
        art = f"第{n}条"
    except ValueError:
        art = f"第{art_id_str}条"

    return f"{base}{art}"


ARTICLES: List[Dict[str, Any]] = load_articles(CSV_PATH)


# ========= 検索ロジック =========

def bigrams(s: str) -> List[str]:
    """
    日本語テキストを2文字ずつのバイグラムに分割。
    """
    s = s.replace(" ", "").replace("\n", "")
    if not s:
        return []
    if len(s) == 1:
        return [s]
    return [s[i : i + 2] for i in range(len(s) - 1)]


def search_articles(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    自然文クエリを、バイグラムの重なりで ARTICLES から検索。
    keywords + heading をやや重く評価。
    """
    q_bi = set(bigrams(query))
    if not q_bi:
        return []

    scored: List[tuple[float, Dict[str, Any]]] = []

    for art in ARTICLES:
        kw_text = f"{art.get('keywords', '')} {art.get('heading', '')}"
        body = art.get("text", "")

        kw_bi = set(bigrams(kw_text))
        body_bi = set(bigrams(body))
        
        inter_kw = q_bi & kw_bi
        inter_body = q_bi & body_bi

        if not inter_kw and not inter_body:
            continue

        score = (2 * len(inter_kw) + len(inter_body)) / max(1, len(q_bi))
        scored.append((score, art))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [art for score, art in scored[:top_k]]


# ========= LLM 応答生成 =========

def build_system_prompt() -> str:
    """
    AI理事長としての振る舞いを定義するシステムメッセージ。
    """
    return """あなたは分譲マンション「プレサンスロジェ岐阜長良橋通り管理組合法人」のAI管理人です。
以下のルールに従って住民からの質問に回答してください。

・必ず管理規約・各種細則などの条文を根拠として回答すること。
・条文を引用する際は、正式名称で示すこと。
  例：「管理規約第◯条」「ペット飼育細則第◯条」「駐車場使用細則第◯条」など。
・根拠となる条文名と条番号を明示し、その内容を住民にも分かるように噛み砕いて説明すること。
・口調は、柔らかく丁寧だが、管理人として責任ある落ち着いた文体とすること。
・条文が明確に禁止／許可していない場合は、その旨を率直に伝え、「理事会への相談」や
  「管理会社を通じた確認」など、現実的な次の一歩を提案すること。
・法律専門家としての最終的な法的判断ではなく、「管理組合としての運用上の考え方」を示す前提で答えること。
【重要】
・前の質問内容を推測したり、参照したりしてはいけません。
・現在のユーザー入力のみを根拠に回答しなさい。
・他の話題（排尿・ペット・騒音など）を持ち出してはいけません。
・質問と無関係な論点を勝手に混ぜてはいけません。
・違う話題に切り替わった場合、前の話題は完全に無視しなさい。

"""


def build_context_block(articles: List[Dict[str, Any]]) -> str:
    """
    LLM に渡すコンテキスト（条文一覧）を組み立てる。
    """
    lines = []
    for idx, art in enumerate(articles, start=1):
        label = art["label"]
        heading = art.get("heading", "")
        text = art.get("text", "")
        lines.append(f"[{idx}] {label} {heading}\n{text}\n")
    return "\n---\n".join(lines)


def generate_answer_with_llm(question: str, articles: List[Dict[str, Any]]) -> str:
    """
    OpenAI API を用いて、条文コンテキスト付きで回答を生成する。
    住民向けに人間味のあるトーンで返答し、そのうえで関連する条文を
    必要な範囲だけ抜粋して示す。
    """
    if client is None:
        # APIキー未設定などでクライアントがない場合は LLM を使わず終了
        return ""

    # --- 条文コンテキストを短く整形（AI の参考用） ---
    context_blocks: List[str] = []

    for i, art in enumerate(articles, start=1):
        # load_articles で入っている実際のキーを使う
        label = art.get("label", "")        # 例：管理規約第◯条
        heading = art.get("heading", "")    # 見出し
        body = art.get("text", "")          # 条文本文

        header = f"{label} {heading}".strip() or f"候補 {i}"

        body_str = (body or "").strip()
        if len(body_str) > 300:
            body_str = body_str[:300] + "……"

        block = f"■ {header}\n{body_str}"
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)

    # --- AI管理人の人格設定（build_system_prompt を呼び出す） ---
    system_prompt = build_system_prompt()

    # --- ユーザーメッセージ ---
    user_prompt = f"""
住民から次の問い合わせがありました。これに対する返信文を作成してください。

【住民からのメッセージ】
{question}

【参考となる規約・細則の条文（検索結果の抜粋）】
{context_text}

上記の条文を参考にしつつ、住民に寄り添う形で回答してください。
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # ここを gpt-4.1 に統一
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=700,
        )
        return "【LLM回答モード】\n" + (resp.choices[0].message.content or "").strip()
    except Exception as e:
        # 本番運用ではログだけ出して、呼び出し元でフォールバックに回す
        print(f"LLM error: {e}")
        return ""




def build_fallback_answer(question: str, articles: List[Dict[str, Any]]) -> str:
    ###############################################
    # 連絡先表示ロジック（管理会社＋タワーパーキング）
    ###############################################

    def get_management_company_section() -> str:
        """管理会社に連絡すべき案件の場合に表示される文面"""
        return (
            "【管理会社連絡先】\n"
            "日本管財住宅管理株式会社\n"
            "平日（9:00～18:00）：052-857-0051\n"
            "上記以外および土日祝・年末年始（12/30～1/3）：0120-051-505"
        )

    def get_tower_parking_section(now: datetime | None = None) -> str:
        """タワーパーキング案件のときに表示される文面"""
        if now is None:
            # Render の UTC から日本時間に変換（UTC+9）
            now = datetime.utcnow() + timedelta(hours=9)

        weekday = now.weekday()  # 月=0 〜 日=6
        hour = now.hour

        is_sunday = (weekday == 6)
        is_wednesday = (weekday == 2)

        # 管理員 在室条件：10〜17時 & 日曜/水曜以外
        manager_available = (10 <= hour < 17) and not (is_sunday or is_wednesday)

        lines: list[str] = []
        lines.append("【タワーパーキングに関する連絡先】")

        if manager_available:
            lines.append(
                "・現在は管理員在室時間帯（10:00～17:00・日/水を除く）です。まず管理員室へお声がけください。"
            )
        else:
            lines.append(
                "・現在は管理員不在時間帯です。以下のサービスセンターへご連絡ください。"
            )

        lines.append("　ＩＨＩ運搬機械（株）岐阜サービスセンター")
        lines.append("　電話番号：058-268-3380")
        lines.append("")
        lines.append(
            "※タワーパーキングのドアが開いたままで次の方の車が出庫できない等の場合、"
            "管理組合のLINE公式アカウントよりご連絡ください。"
            "マスターキーの保管場所をご案内できる場合があります。"
        )

        return "\n".join(lines)

    def append_contact_sections(answer_text: str, original_question: str) -> str:
        """
        LLMやフォールバックで組み立てた回答に対して、
        ・回答文の中に「管理会社」が出てきたら → 管理会社の電話番号を追記
        ・質問 or 回答の中にタワーパーキング関連ワードがあれば → タワーパーキング連絡先を追記
        という形で末尾に連絡先案内を自動付与します。
        """
        text_for_detection = (answer_text or "") + "\n" + (original_question or "")

        # 1) 管理会社への連絡が必要そうな回答 → 回答文中に「管理会社」が出てきたら追記
        if "管理会社" in answer_text:
            answer_text += "\n\n" + get_management_company_section()

        # 2) タワーパーキング関連ワードの検知
        tower_keywords = ["タワーパーキング", "立体駐車場", "タワー駐車場"]
        if any(kw in text_for_detection for kw in tower_keywords):
            answer_text += "\n\n" + get_tower_parking_section()

        return answer_text

    """
    OpenAI が使えない場合の簡易回答。
    条文を列挙するだけ。
    """
    if not articles:
        return "該当しそうな規約・細則の条文を見つけられませんでした。理事会または管理会社へご相談ください。"

    lines = [
        "【フォールバック規約抜粋モード】",
        "関連がありそうな条文を抜粋します。詳細な解釈は理事会での検討が必要です。\n",
    ]
    for art in articles[:3]:
        lines.append(f"■ {art['label']} {art.get('heading', '')}")
        lines.append(art.get("text", ""))
        lines.append("")
    return "\n".join(lines).strip()



def answer_question_text(question: str) -> str:
    def answer_question_text(question: str) -> str:
        """
        生テキストの質問に対して、回答文だけ返すユーティリティ。
        （/ask API と LINE Webhook の両方から利用）
        """
        q = question.strip()
        if not q:
            return "質問の内容が空でした。もう一度入力してください。"

        matched = search_articles(q, top_k=5)

        if matched:
            ans = generate_answer_with_llm(q, matched)
            if not ans:
                ans = build_fallback_answer(q, matched)
        else:
            ans = "該当しそうな規約・細則の条文を見つけられませんでした。理事会または管理会社へご相談ください。"

        # ★ 管理会社／タワーパーキングの連絡先を自動で追記
        ans = append_contact_sections(ans, q)

        return ans


# ========= FastAPI 定義 =========

class QuestionRequest(BaseModel):
    question: str


class ReferenceItem(BaseModel):
    kind: str
    id: str
    label: str
    heading: str


class AnswerResponse(BaseModel):
    answer: str
    references: List[ReferenceItem]


app = FastAPI(
    title="AI理事長 API",
    description="プレサンスロジェ岐阜長良橋通り管理組合法人向け AI理事長エンジン",
)

# CORS（フロントが別Originの場合を想定）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要に応じて絞り込み可
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(payload: QuestionRequest):
    """
    HTTP API 用：住民からの質問（日本語自然文）を受け取り、
    規約・細則を検索して回答を返すエンドポイント。
    """
    question = payload.question.strip()
    if not question:
        return AnswerResponse(answer="質問の内容が空でした。もう一度入力してください。", references=[])

    matched = search_articles(question, top_k=5)

    if matched:
        answer = generate_answer_with_llm(question, matched)
        if not answer:
            answer = build_fallback_answer(question, matched)
    else:
        answer = "該当しそうな規約・細則の条文を見つけられませんでした。理事会または管理会社へご相談ください。"

    refs = [
        ReferenceItem(
            kind=art["kind"],
            id=str(art["id"]),
            label=art["label"],
            heading=art.get("heading", ""),
        )
        for art in matched
    ]

    return AnswerResponse(answer=answer, references=refs)


@app.get("/health")
async def health_check():
    return {"status": "ok", "articles": len(ARTICLES)}


# ========= LINE Webhook =========

@app.post("/webhook")
async def line_webhook(request: Request):
    """
    LINE からの Webhook を受け取り、テキストメッセージには AI理事長の回答を返す。
    """
    if not LINE_CHANNEL_SECRET or not LINE_CHANNEL_ACCESS_TOKEN:
        raise HTTPException(status_code=500, detail="LINEチャンネル情報が設定されていません。")

    signature = request.headers.get("x-line-signature")
    body_bytes = await request.body()
    body = body_bytes.decode("utf-8")

    if signature is None:
        raise HTTPException(status_code=400, detail="X-Line-Signature header missing")

    try:
        events = line_parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    with ApiClient(line_config) as api_client:
        messaging_api = MessagingApi(api_client)

        for event in events:
            if isinstance(event, MessageEvent) and isinstance(event.message, TextMessageContent):
                user_text = event.message.text or ""
                reply_text = answer_question_text(user_text)

                reply_request = ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply_text)],
                )
                messaging_api.reply_message(reply_request)

    return "OK"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
