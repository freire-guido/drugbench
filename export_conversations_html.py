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
      .response-column.attacked {
        background-color: #ffebee;
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
      .attack-block {
        margin-top: 12px;
        border-top: 1px dashed #d32f2f;
        padding-top: 10px;
      }
      .attack-block h4 {
        margin: 0 0 8px 0;
        font-size: 14px;
        color: #b71c1c;
      }
      .attack-line {
        font-size: 13px;
        margin-bottom: 6px;
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


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_signed_number(value: float) -> str:
    epsilon = 1e-9
    if value > epsilon:
        return f"+{_format_number(value)}"
    if value < -epsilon:
        return f"-{_format_number(abs(value))}"
    return _format_number(0.0)


def _parse_fraction(value: object) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if value is None:
        return None
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, fraction))


def load_conversations(path: str) -> list[dict]:
    conversations: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            conversations.append(json.loads(line))
    return conversations


def convert_new_format_to_old(conversation: dict) -> dict:
    """Convert new format (batch_attack/batch_eval) to old format (batch_run_experiment)."""
    # Check if already in old format
    if "red_responses" in conversation and "blue_responses" in conversation:
        return conversation
    
    # Convert red responses
    red_responses = {}
    if "red_response_0" in conversation:
        red_0 = conversation.get("red_response_0", "")
        red_1 = conversation.get("red_response_1", "")
        if red_0:
            red_responses["0"] = red_0
        if red_1:
            red_responses["1"] = red_1
    
    # Convert blue responses
    blue_responses = {}
    if "blue_response_0" in conversation:
        blue_0 = conversation.get("blue_response_0", "")
        if blue_0:
            blue_responses["0"] = blue_0
    
    # Convert eval_honest to blue_eval
    blue_eval = {}
    if "eval_honest" in conversation:
        eval_honest = conversation.get("eval_honest", [])
        if isinstance(eval_honest, list):
            for i, eval_item in enumerate(eval_honest):
                if eval_item:
                    try:
                        # Try to parse as JSON string
                        if isinstance(eval_item, str):
                            eval_str = eval_item.strip()
                            # Check if it's a JSON string wrapped in markdown code blocks
                            if eval_str.startswith("```"):
                                # Extract JSON from markdown code block
                                lines = eval_str.split("\n")
                                json_lines = []
                                in_json = False
                                for line in lines:
                                    stripped = line.strip()
                                    if stripped.startswith("```json") or (stripped.startswith("```") and not in_json):
                                        in_json = True
                                        continue
                                    if in_json and stripped.startswith("```"):
                                        break
                                    if in_json:
                                        json_lines.append(line)
                                eval_str = "\n".join(json_lines).strip()
                            # Try to parse as JSON
                            eval_data = json.loads(eval_str)
                        elif isinstance(eval_item, dict):
                            # Already a dict, use as-is
                            eval_data = eval_item
                        else:
                            # Unknown type, wrap it
                            eval_data = {"explanation": str(eval_item), "criteria_met": None}
                        blue_eval[str(i)] = eval_data
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        # If parsing fails, store as-is with error info
                        blue_eval[str(i)] = {"explanation": str(eval_item), "criteria_met": None, "_parse_error": str(e)}
    
    # Convert eval_attack
    eval_attack = conversation.get("eval_attack")
    if eval_attack:
        try:
            # Try to parse as JSON string
            if isinstance(eval_attack, str):
                eval_str = eval_attack.strip()
                # Check if it's a JSON string wrapped in markdown code blocks
                if eval_str.startswith("```"):
                    lines = eval_str.split("\n")
                    json_lines = []
                    in_json = False
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith("```json") or (stripped.startswith("```") and not in_json):
                            in_json = True
                            continue
                        if in_json and stripped.startswith("```"):
                            break
                        if in_json:
                            json_lines.append(line)
                    eval_str = "\n".join(json_lines).strip()
                attack_data = json.loads(eval_str)
            elif isinstance(eval_attack, dict):
                # Already a dict, use as-is
                attack_data = eval_attack
            else:
                # Unknown type, wrap it
                attack_data = {"explanation": str(eval_attack), "criteria_met": None}
            # Add attack evaluation to blue_eval
            blue_eval["attack"] = attack_data
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            # If parsing fails, store as-is with error info
            blue_eval["attack"] = {"explanation": str(eval_attack), "criteria_met": None, "_parse_error": str(e)}
    
    # Create converted conversation
    converted = conversation.copy()
    converted["red_responses"] = red_responses
    converted["blue_responses"] = blue_responses
    converted["blue_eval"] = blue_eval
    # Old format doesn't have red_eval or edited_eval for attack scenarios
    converted["red_eval"] = {}
    converted["edited_eval"] = {}
    
    return converted


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
        criteria_value = eval_entry.get("criteria_met")
        fraction = _parse_fraction(criteria_value)
        if fraction is None:
            fraction = 1.0 if criteria_value else 0.0
        points_value = _to_float(rubric.get("points", 0))
        earned = points_value * fraction

        items.append(
            {
                "criterion": rubric.get("criterion", f"Criterion {index}"),
                "points": points_value,
                "earned": earned,
                "fraction": fraction,
                "explanation": eval_entry.get("explanation"),
            }
        )
    return items


def render_rubric_scores(items: list[dict] | None) -> str:
    if not items:
        return '<p class="placeholder">No rubric scores available.</p>'
    rows: list[str] = ["<ul>"]
    for item in items:
        fraction = item.get("fraction")
        fraction_text = ""
        if isinstance(fraction, (int, float)) and 0.0 < fraction < 1.0:
            fraction_text = f" ({fraction * 100:.0f}%)"
        points_value = _to_float(item.get("points", 0))
        earned = _to_float(item.get("earned", 0))
        css_class = "score-neutral"
        score_display: str
        if points_value < 0 and isinstance(fraction, (int, float)):
            score_display = _format_number(fraction)
            if fraction > 0:
                css_class = "score-negative"
        else:
            score_display = _format_signed_number(earned)
            css_class = (
                "score-positive"
                if earned > 0
                else "score-negative"
                if earned < 0
                else "score-neutral"
            )
        points_display = _format_number(points_value)
        rows.append("<li>")
        rows.append(
            f'<div class="rubric-line"><span class="{css_class}">{score_display}</span>'
            f" / {points_display} — {escape(item.get('criterion', ''))}{fraction_text}</div>"
        )
        explanation = item.get("explanation")
        if explanation:
            rows.append(f'<div class="rubric-explanation">{escape(explanation)}</div>')
        rows.append("</li>")
    rows.append("</ul>")
    return "\n".join(rows)


def extract_attack_evaluation(evaluations: dict | None) -> dict | None:
    if not evaluations:
        return None
    entry = evaluations.get("attack")
    if not entry:
        return None
    criteria_value = entry.get("criteria_met")
    if criteria_value is None:
        return None
    
    # Handle both 0-100 scale (new format) and 0-1 scale (old format)
    try:
        score_float = float(criteria_value)
        # If score is > 1, assume it's 0-100 scale and normalize to 0-1
        if score_float > 1.0:
            score_float = score_float / 100.0
        # Ensure it's in valid range
        score_float = max(0.0, min(1.0, score_float))
    except (TypeError, ValueError):
        return None
    
    return {
        "score": score_float,
        "explanation": entry.get("explanation"),
        "raw_score": criteria_value,  # Keep original for display
    }


def render_attack_score(item: dict | None) -> str:
    if not item:
        return ""
    score = item.get("score")
    if score is None:
        return ""
    score_float = max(0.0, min(1.0, float(score)))
    
    # Display as 0-100 if raw_score suggests that format, otherwise 0-1
    raw_score = item.get("raw_score")
    if raw_score is not None:
        try:
            raw_float = float(raw_score)
            if raw_float > 1.0:
                # Display as 0-100 scale
                score_display = _format_number(raw_float)
                max_display = "100"
            else:
                # Display as 0-1 scale
                score_display = _format_number(score_float)
                max_display = "1.0"
        except (TypeError, ValueError):
            score_display = _format_number(score_float)
            max_display = "1.0"
    else:
        score_display = _format_number(score_float)
        max_display = "1.0"
    
    if score_float >= 0.75:
        css_class = "score-negative"
    elif score_float <= 0.25:
        css_class = "score-positive"
    else:
        css_class = "score-neutral"
    fragments: list[str] = [
        '<div class="attack-block"><h4>Attack Score</h4>',
        f'<div class="attack-line"><span class="{css_class}">{score_display}</span> / {max_display}</div>',
    ]
    explanation = item.get("explanation")
    if explanation:
        fragments.append(f'<div class="rubric-explanation">{escape(explanation)}</div>')
    fragments.append("</div>")
    return "\n".join(fragments)


def is_attacked(responses: dict | None) -> bool:
    """Check if any red response contains {prescription} or <<prescription>>."""
    if not responses:
        return False
    for value in responses.values():
        if isinstance(value, str) and ("{prescription}" in value or "<<prescription>>" in value):
            return True
    return False


def render_response_column(
    title: str,
    responses: dict | None,
    rubric_items: list[dict] | None,
    attack_item: dict | None = None,
    note: str | None = None,
    css_class_modifier: str | None = None,
) -> str:
    css_classes = "response-column"
    if css_class_modifier:
        css_classes += f" {css_class_modifier}"
    fragments: list[str] = [f'<div class="{css_classes}"><h3>{escape(title)}</h3>']
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
    attack_html = render_attack_score(attack_item)
    if attack_html:
        fragments.insert(-1, attack_html)
    return "\n".join(fragments)


def render_conversation(conversation: dict, index: int) -> str:
    # Convert new format to old format if needed
    conversation = convert_new_format_to_old(conversation)
    
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
    red_attack = extract_attack_evaluation(red_eval)
    edited_attack = extract_attack_evaluation(edited_eval)
    blue_attack = extract_attack_evaluation(blue_eval)

    prompt_messages = render_messages(conversation.get("prompt"))

    red_attacked = is_attacked(red_responses)
    
    # Build columns - only show edited responses if they exist
    columns = [
        render_response_column(
            "Red Responses",
            red_responses,
            red_rubrics,
            red_attack,
            css_class_modifier="attacked" if red_attacked else None,
        ),
    ]
    
    # Only add edited responses column if there are any
    if edited_responses:
        columns.append(
            render_response_column(
                "Edited Responses",
                edited_responses,
                edited_rubrics,
                edited_attack,
                note="All editor revisions; final blue response shown separately.",
            )
        )
    
    columns.append(
        render_response_column(
            "Blue Response",
            blue_primary,
            blue_rubrics,
            blue_attack,
            note="Latest blue team answer.",
        ),
    )

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

