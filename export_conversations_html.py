import argparse
import html
import json
from string import Template

PAGE_TEMPLATE = Template("""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Conversation Report</title>
    <style>
      body {
        font-family: Arial, sans-serif;
        margin: 24px;
        background-color: #f7f7f7;
        color: #212121;
      }
      .conversation {
        background: #ffffff;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 24px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06);
      }
      .conversation header {
        margin-bottom: 12px;
      }
      .conversation header h2 {
        margin: 0 0 4px 0;
        font-size: 20px;
      }
      .conversation header .tags {
        font-size: 12px;
        color: #555555;
      }
      .messages {
        margin-bottom: 16px;
      }
      .message {
        border-left: 4px solid #2196f3;
        padding: 8px 12px;
        margin-bottom: 8px;
        background-color: #f1f9ff;
      }
      .message.assistant {
        border-color: #9c27b0;
        background-color: #f6e8fb;
      }
      .message.system {
        border-color: #607d8b;
        background-color: #eceff1;
      }
      .message .role {
        font-weight: bold;
        margin-bottom: 4px;
      }
      .responses {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
      }
      .response-column {
        flex: 1 1 300px;
        background-color: #fbfbfb;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 12px 14px;
      }
      .response-column h3 {
        margin-top: 0;
      }
      .response-column .note {
        font-size: 12px;
        color: #666666;
        margin-bottom: 8px;
      }
      .response-item {
        margin-bottom: 12px;
      }
      .response-label {
        font-size: 12px;
        font-weight: bold;
        color: #666666;
        margin-bottom: 4px;
      }
      .response-item pre {
        white-space: pre-wrap;
        word-break: break-word;
        margin: 0;
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 8px;
      }
      .placeholder {
        color: #9e9e9e;
        font-style: italic;
      }
      .rubric-block {
        margin-top: 12px;
        border-top: 1px solid #e0e0e0;
        padding-top: 10px;
      }
      .rubric-block h4 {
        margin: 0 0 8px 0;
        font-size: 14px;
      }
      .rubric-block ul {
        list-style-type: none;
        padding-left: 0;
        margin: 0;
      }
      .rubric-block li {
        margin-bottom: 6px;
        font-size: 13px;
      }
      .rubric-line {
        margin-bottom: 4px;
      }
      .rubric-explanation {
        margin-left: 18px;
        color: #555555;
        font-size: 12px;
        white-space: pre-wrap;
      }
      .score-positive {
        color: #1b5e20;
      }
      .score-negative {
        color: #b71c1c;
      }
      .score-neutral {
        color: #424242;
      }
    </style>
  </head>
  <body>
    <main>
      $content
    </main>
  </body>
</html>
""")


def load_conversations(path: str) -> list[dict]:
    conversations: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            conversations.append(json.loads(line))
    return conversations


def escape(text: str) -> str:
    if text is None:
        return ""
    return html.escape(str(text))


def render_messages(messages: list[dict]) -> str:
    fragments: list[str] = []
    for message in messages or []:
        role = message.get("role", "unknown")
        css_class = f"message {role}"
        variant = message.get("variant")
        role_label = role.title()
        if variant:
            role_label = f"{role_label} ({variant})"
        fragments.append(
            f'<div class="{css_class}">'
            f'<div class="role">{escape(role_label)}</div>'
            f"<pre>{escape(message.get('content', ''))}</pre>"
            "</div>"
        )
    if not fragments:
        return '<p class="placeholder">No prompt messages available.</p>'
    return "\n".join(fragments)


def compute_rubric_items(
    rubrics: list[dict],
    evaluations: dict | None,
) -> list[dict]:
    evaluations = evaluations or {}
    items: list[dict] = []
    for index, rubric in enumerate(rubrics or []):
        eval_entry = evaluations.get(str(index)) or {}
        criteria_met = bool(eval_entry.get("criteria_met"))
        points = rubric.get("points", 0)
        earned = points if criteria_met else 0
        items.append(
            {
                "criterion": rubric.get("criterion", f"Criterion {index}"),
                "points": points,
                "earned": earned,
                "explanation": eval_entry.get("explanation"),
            }
        )
    return items


def render_rubric_scores(items: list[dict] | None) -> str:
    if not items:
        return '<p class="placeholder">No rubric scores available.</p>'
    rows: list[str] = ["<ul>"]
    for item in items:
        earned = item.get("earned", 0)
        css_class = (
            "score-positive"
            if earned > 0
            else "score-negative"
            if earned < 0
            else "score-neutral"
        )
        sign = "+" if earned > 0 else ""
        rows.append("<li>")
        rows.append(
            f'<div class="rubric-line"><span class="{css_class}">{sign}{earned}</span>'
            f" / {item.get('points', 0)} — {escape(item.get('criterion', ''))}</div>"
        )
        explanation = item.get("explanation")
        if explanation:
            rows.append(f'<div class="rubric-explanation">{escape(explanation)}</div>')
        rows.append("</li>")
    rows.append("</ul>")
    return "\n".join(rows)


def render_response_column(
    title: str,
    responses: dict | None,
    rubric_items: list[dict] | None,
    note: str | None = None,
) -> str:
    fragments: list[str] = [f'<div class="response-column"><h3>{escape(title)}</h3>']
    if note:
        fragments.append(f'<p class="note">{escape(note)}</p>')
    has_responses = False
    for key, value in responses.items() if responses else []:
        has_responses = True
        fragments.append(
            '<div class="response-item">'
            f'<div class="response-label">Variant {escape(key)}</div>'
            f"<pre>{escape(value)}</pre>"
            "</div>"
        )
    if not has_responses:
        fragments.append('<div class="response-item placeholder">No responses available.</div>')
    fragments.append('<div class="rubric-block"><h4>Rubric Scores</h4>')
    fragments.append(render_rubric_scores(rubric_items))
    fragments.append("</div></div>")
    return "\n".join(fragments)


def render_conversation(conversation: dict, index: int) -> str:
    prompt_id = conversation.get("prompt_id", f"conversation-{index}")
    tags = ", ".join(conversation.get("example_tags", []))
    red_responses = conversation.get("red_responses") or {}
    blue_responses = conversation.get("blue_responses") or {}
    edited_responses, blue_primary = split_blue_responses(blue_responses)

    rubrics = conversation.get("rubrics") or []
    red_eval = conversation.get("red_eval") or {}
    edited_eval = conversation.get("edited_eval") or {}
    blue_eval = conversation.get("blue_eval") or {}

    red_rubrics = compute_rubric_items(rubrics, red_eval)
    edited_rubrics = compute_rubric_items(rubrics, edited_eval)
    blue_rubrics = compute_rubric_items(rubrics, blue_eval)

    prompt_messages = render_messages(conversation.get("prompt"))

    columns = [
        render_response_column("Red Responses", red_responses, red_rubrics),
        render_response_column(
            "Edited Responses",
            edited_responses,
            edited_rubrics,
            note="All editor revisions; final blue response shown separately.",
        ),
        render_response_column(
            "Blue Response",
            blue_primary,
            blue_rubrics,
            note="Latest blue team answer.",
        ),
    ]

    return (
        '<section class="conversation">'
        "<header>"
        f"<h2>{escape(prompt_id)}</h2>"
        f'<div class="tags">{escape(tags) if tags else "No tags"}</div>'
        "</header>"
        '<div class="messages">'
        f"{prompt_messages}"
        "</div>"
        '<div class="responses">'
        f'{"".join(columns)}'
        "</div>"
        "</section>"
    )


def build_report(conversations: list[dict]) -> str:
    sections = [
        render_conversation(conversation, index + 1)
        for index, conversation in enumerate(conversations)
    ]
    return PAGE_TEMPLATE.substitute(content="\n".join(sections))


def split_blue_responses(responses: dict) -> tuple[dict, dict]:
    if not responses:
        return {}, {}
    items = list(responses.items())
    blue_key, blue_value = items[-1]
    edited_items = items[:-1]
    edited = dict(edited_items)
    blue = {blue_key: blue_value}
    return edited, blue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export conversations JSONL into an HTML report."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the input conversations JSONL file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the generated HTML report.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()
    if args.verbose:
        print(f"Loading conversations from {args.file}")
    conversations = load_conversations(args.file)

    if args.output is None:
        args.output = args.file.replace('.jsonl', '.html')

    if args.verbose:
        print(f"Preparing report for {len(conversations)} conversations")

    html_report = build_report(conversations)
    with open(args.output, "w+", encoding="utf-8") as handle:
        handle.write(html_report)

    if args.verbose:
        print(f"Report written to {args.output}")
    else:
        print(f"Wrote {len(conversations)} conversations to {args.output}")


if __name__ == "__main__":
    main()

