#!/usr/bin/env python3
"""Convert conversations JSONL files into readable HTML reports."""

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
      :root {
        --bg: #0d1117;
        --card-bg: #161b22;
        --border: #30363d;
        --text: #e6edf3;
        --text-muted: #8b949e;
        --accent-blue: #58a6ff;
        --accent-red: #f85149;
        --accent-green: #3fb950;
        --accent-purple: #a371f7;
        --accent-orange: #d29922;
        --user-bg: #1c2128;
        --assistant-bg: #1a1f29;
        --system-bg: #21262d;
        --attack-bg: #3d1a1a;
        --attack-border: #f8514966;
        --attack-failed-bg: #3d3a1a;
        --attack-failed-border: #d2992266;
      }

      * {
        box-sizing: border-box;
      }

      body {
        font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace;
        margin: 0;
        padding: 24px;
        background: var(--bg);
        color: var(--text);
        line-height: 1.5;
      }

      h1 {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0 0 24px 0;
        color: var(--text);
        border-bottom: 1px solid var(--border);
        padding-bottom: 16px;
      }

      .conversation-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        margin-bottom: 32px;
        overflow: hidden;
      }

      .card-header {
        padding: 16px 20px;
        border-bottom: 1px solid var(--border);
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        flex-wrap: wrap;
        gap: 12px;
      }

      .prompt-id {
        font-size: 1rem;
        font-weight: 600;
        color: var(--accent-blue);
        word-break: break-all;
      }

      .tags {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }

      .tag {
        font-size: 0.7rem;
        padding: 2px 8px;
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        color: var(--text-muted);
      }

      .expected-harm-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
      }

      .expected-harm-badge.low {
        background: #1a3d1a;
        border: 1px solid var(--accent-green);
        color: var(--accent-green);
      }

      .expected-harm-badge.medium {
        background: #3d3a1a;
        border: 1px solid var(--accent-orange);
        color: var(--accent-orange);
      }

      .expected-harm-badge.high {
        background: #3d1a1a;
        border: 1px solid var(--accent-red);
        color: var(--accent-red);
      }

      .conversation-section {
        padding: 16px 20px;
        border-bottom: 1px solid var(--border);
      }

      .section-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        margin-bottom: 12px;
      }

      .messages {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }

      .message {
        padding: 10px 14px;
        border-radius: 6px;
        border-left: 3px solid;
      }

      .message.user {
        background: var(--user-bg);
        border-color: var(--accent-blue);
      }

      .message.assistant {
        background: var(--assistant-bg);
        border-color: var(--accent-purple);
      }

      .message.system {
        background: var(--system-bg);
        border-color: var(--text-muted);
      }

      .message-role {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        margin-bottom: 4px;
      }

      .message-content {
        white-space: pre-wrap;
        word-break: break-word;
        font-size: 0.9rem;
      }

      .responses-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 0;
      }

      @media (max-width: 1200px) {
        .responses-grid {
          grid-template-columns: 1fr;
        }
      }

      .response-column {
        padding: 16px 20px;
        border-right: 1px solid var(--border);
      }

      .response-column:last-child {
        border-right: none;
      }

      .response-column.attacked {
        background: var(--attack-bg);
        border-color: var(--attack-border);
      }

      .response-column.attacked-failed {
        background: var(--attack-failed-bg);
        border-color: var(--attack-failed-border);
      }

      .column-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
      }

      .column-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--accent-red);
      }

      .response-column.blue-edit .column-title {
        color: var(--accent-blue);
      }

      .response-column.blue-defer .column-title {
        color: var(--accent-purple);
      }

      .attack-badge {
        font-size: 0.7rem;
        padding: 2px 8px;
        background: var(--accent-red);
        color: var(--bg);
        border-radius: 4px;
        font-weight: 600;
      }

      .attack-badge.failed {
        background: var(--accent-orange);
      }

      .response-block {
        margin-bottom: 16px;
      }

      .response-block:last-child {
        margin-bottom: 0;
      }

      .response-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .response-label .prescription-marker {
        color: var(--accent-red);
        font-weight: 600;
      }

      .response-text {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 12px;
        white-space: pre-wrap;
        word-break: break-word;
        font-size: 0.85rem;
        max-height: 400px;
        overflow-y: auto;
      }

      .response-text.empty {
        color: var(--text-muted);
        font-style: italic;
      }

      .eval-section {
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px dashed var(--border);
      }

      .eval-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        margin-bottom: 6px;
      }

      .eval-content {
        font-size: 0.8rem;
        color: var(--text-muted);
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 8px;
        white-space: pre-wrap;
        word-break: break-word;
      }

      .rubrics-section {
        padding: 16px 20px;
        border-top: 1px solid var(--border);
      }

      .rubric-item {
        margin-bottom: 8px;
        padding: 8px 12px;
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 4px;
      }

      .rubric-item:last-child {
        margin-bottom: 0;
      }

      .rubric-criterion {
        font-size: 0.85rem;
        margin-bottom: 4px;
      }

      .rubric-meta {
        font-size: 0.75rem;
        color: var(--text-muted);
      }

      .placeholder {
        color: var(--text-muted);
        font-style: italic;
      }

      .logprobs-info {
        font-size: 0.7rem;
        color: var(--text-muted);
        margin-top: 4px;
      }
    </style>
  </head>
  <body>
    <h1>Conversation Report</h1>
    $content
  </body>
</html>
""")


def escape(text: str | None) -> str:
    """HTML-escape text, handling None values."""
    if text is None:
        return ""
    return html.escape(str(text))


def load_conversations(path: str) -> list[dict]:
    """Load conversations from a JSONL file."""
    conversations: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            conversations.append(json.loads(line))
    return conversations


def get_attack_status(conversation: dict) -> str:
    """Check attack status based on {prescription} marker presence.
    
    Returns:
        'success' - {prescription} present and red_0 has content (successful attack)
        'failed' - {prescription} present but red_0 is empty (unsuccessful attack)
        'none' - no {prescription} marker found
    """
    red_0 = conversation.get("red_0", "")
    red_1 = conversation.get("red_1", "")
    has_prescription = "{prescription}" in str(red_0) or "{prescription}" in str(red_1)
    
    if not has_prescription:
        return "none"
    
    # Check if red_0 has actual content (successful attack)
    if red_0 and red_0.strip():
        return "success"
    else:
        return "failed"


def get_harm_class(blue_0: str | None) -> str:
    """Get CSS class based on expected harm value."""
    if blue_0 is None:
        return "low"
    try:
        value = int(blue_0)
        if value <= 25:
            return "low"
        elif value <= 50:
            return "medium"
        else:
            return "high"
    except (ValueError, TypeError):
        return "low"


def render_messages(messages: list[dict]) -> str:
    """Render conversation messages as HTML."""
    if not messages:
        return '<p class="placeholder">No conversation messages.</p>'
    
    fragments: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        css_class = f"message {role}"
        
        fragments.append(f'''
        <div class="{css_class}">
          <div class="message-role">{escape(role)}</div>
          <div class="message-content">{escape(content)}</div>
        </div>
        ''')
    
    return "\n".join(fragments)


def render_response_block(label: str, content: str | None, has_prescription: bool = False) -> str:
    """Render a single response block."""
    prescription_marker = ""
    if has_prescription:
        prescription_marker = '<span class="prescription-marker">[contains {prescription}]</span>'
    
    if not content:
        return f'''
        <div class="response-block">
          <div class="response-label">{escape(label)} {prescription_marker}</div>
          <div class="response-text empty">Not available</div>
        </div>
        '''
    
    return f'''
    <div class="response-block">
      <div class="response-label">{escape(label)} {prescription_marker}</div>
      <div class="response-text">{escape(content)}</div>
    </div>
    '''


def render_eval_block(label: str, eval_data: str | list | dict | None) -> str:
    """Render an evaluation block."""
    if not eval_data:
        return ""
    
    # Format the eval data nicely
    if isinstance(eval_data, (dict, list)):
        content = json.dumps(eval_data, indent=2, ensure_ascii=False)
    else:
        content = str(eval_data)
    
    return f'''
    <div class="eval-section">
      <div class="eval-label">{escape(label)}</div>
      <div class="eval-content">{escape(content)}</div>
    </div>
    '''


def render_expected_harm_badge(blue_0: str | None, logprobs: dict | None = None) -> str:
    """Render the expected harm badge with optional logprobs info."""
    if blue_0 is None:
        return ""
    
    harm_class = get_harm_class(blue_0)
    
    # Extract top logprob info if available
    logprob_info = ""
    if logprobs and "content" in logprobs:
        try:
            top_token = logprobs["content"][0]
            prob = 100 * (2.718281828 ** top_token.get("logprob", 0))
            logprob_info = f'<div class="logprobs-info">confidence: {prob:.1f}%</div>'
        except (IndexError, KeyError, TypeError):
            pass
    
    return f'''
    <div class="expected-harm-badge {harm_class}">
      <span>Expected Harm: {escape(str(blue_0))}</span>
    </div>
    {logprob_info}
    '''


def render_red_column(conversation: dict, attack_status: str) -> str:
    """Render the red team response column."""
    if attack_status == "success":
        css_class = "response-column attacked"
        attack_badge = '<span class="attack-badge">ATTACKED</span>'
    elif attack_status == "failed":
        css_class = "response-column attacked-failed"
        attack_badge = '<span class="attack-badge failed">ATTACK FAILED</span>'
    else:
        css_class = "response-column"
        attack_badge = ""
    
    red_response = conversation.get("red_response", "")
    red_0 = conversation.get("red_0", "")
    red_1 = conversation.get("red_1", "")
    blue_0 = conversation.get("blue_0", "")
    blue_0_logprobs = conversation.get("blue_0_logprobs")
    
    # Check which responses contain {prescription}
    red_0_has_prescription = "{prescription}" in str(red_0)
    red_1_has_prescription = "{prescription}" in str(red_1)
    
    blocks = []
    
    # Expected harm badge
    harm_badge = render_expected_harm_badge(blue_0, blue_0_logprobs)
    if harm_badge:
        blocks.append(f'<div class="response-block">{harm_badge}</div>')
    
    # Main red response
    blocks.append(render_response_block("Red Response", red_response))
    
    # Red 0 if exists
    if red_0:
        blocks.append(render_response_block("Red 0", red_0, red_0_has_prescription))
    
    # Red 1 if exists
    if red_1:
        blocks.append(render_response_block("Red 1", red_1, red_1_has_prescription))
    
    # Evaluations
    eval_attack = conversation.get("eval_red_response_attack")
    eval_honest = conversation.get("eval_red_response_honest")
    
    if eval_attack:
        blocks.append(render_eval_block("Eval (Attack)", eval_attack))
    if eval_honest:
        blocks.append(render_eval_block("Eval (Honest)", eval_honest))
    
    return f'''
    <div class="{css_class}">
      <div class="column-header">
        <div class="column-title">Red Team</div>
        {attack_badge}
      </div>
      {"".join(blocks)}
    </div>
    '''


def render_blue_edit_column(conversation: dict) -> str:
    """Render the blue edit response column."""
    blue_edit = conversation.get("blue_edit", "")
    
    blocks = []
    
    # Blue edit
    blocks.append(render_response_block("Response", blue_edit))
    
    # Evaluations
    eval_edit_attack = conversation.get("eval_blue_edit_attack")
    eval_edit_honest = conversation.get("eval_blue_edit_honest")
    
    if eval_edit_attack:
        blocks.append(render_eval_block("Eval (Attack)", eval_edit_attack))
    if eval_edit_honest:
        blocks.append(render_eval_block("Eval (Honest)", eval_edit_honest))
    
    return f'''
    <div class="response-column blue-edit">
      <div class="column-header">
        <div class="column-title">Blue Edit</div>
      </div>
      {"".join(blocks)}
    </div>
    '''


def render_blue_defer_column(conversation: dict) -> str:
    """Render the blue defer response column."""
    blue_defer = conversation.get("blue_defer", "")
    
    blocks = []
    
    # Blue defer
    blocks.append(render_response_block("Response", blue_defer))
    
    # Evaluations
    eval_defer_attack = conversation.get("eval_blue_defer_attack")
    eval_defer_honest = conversation.get("eval_blue_defer_honest")
    
    if eval_defer_attack:
        blocks.append(render_eval_block("Eval (Attack)", eval_defer_attack))
    if eval_defer_honest:
        blocks.append(render_eval_block("Eval (Honest)", eval_defer_honest))
    
    return f'''
    <div class="response-column blue-defer">
      <div class="column-header">
        <div class="column-title">Blue Defer</div>
      </div>
      {"".join(blocks)}
    </div>
    '''


def render_tags(tags: list[str]) -> str:
    """Render example tags as badges."""
    if not tags:
        return ""
    
    return "".join(f'<span class="tag">{escape(tag)}</span>' for tag in tags)


def render_conversation(conversation: dict, index: int) -> str:
    """Render a single conversation card."""
    prompt_id = conversation.get("prompt_id", f"conversation-{index}")
    example_tags = conversation.get("example_tags", [])
    prompt = conversation.get("prompt", [])
    attack_status = get_attack_status(conversation)
    
    return f'''
    <div class="conversation-card">
      <div class="card-header">
        <div class="prompt-id">{escape(prompt_id)}</div>
        <div class="tags">{render_tags(example_tags)}</div>
      </div>
      
      <div class="conversation-section">
        <div class="section-title">Conversation</div>
        <div class="messages">
          {render_messages(prompt)}
        </div>
      </div>
      
      <div class="responses-grid">
        {render_red_column(conversation, attack_status)}
        {render_blue_edit_column(conversation)}
        {render_blue_defer_column(conversation)}
      </div>
    </div>
    '''


def build_report(conversations: list[dict]) -> str:
    """Build the complete HTML report."""
    sections = [
        render_conversation(conv, i + 1)
        for i, conv in enumerate(conversations)
    ]
    return PAGE_TEMPLATE.substitute(content="\n".join(sections))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert conversations JSONL into an HTML report."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the input conversations JSONL file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the generated HTML report. Defaults to input file with .html extension.",
    )
    args = parser.parse_args()
    
    print(f"Loading conversations from {args.file}")
    conversations = load_conversations(args.file)
    
    if args.output is None:
        args.output = args.file.replace('.jsonl', '.html')
    
    print(f"Building report for {len(conversations)} conversations...")
    html_report = build_report(conversations)
    
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(html_report)
    
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
