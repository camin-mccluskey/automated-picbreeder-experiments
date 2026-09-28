"""Inspection views of selection decisions; never used to score or select images."""

from PIL import Image, ImageDraw


def contact_sheet(images, candidate_ids, decision):
    """Show small measurement sets in full; retain the class summary for ImageNet."""
    width, height, header = 240, 250, 28
    sheet = Image.new("RGB", (3 * width, header + 3 * height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), f"Selection: {decision.mode} | candidate {decision.position + 1}", fill="black")
    for i, (pixels, key) in enumerate(zip(images, candidate_ids)):
        x, y = (i % 3) * width, header + (i // 3) * height
        sheet.paste(Image.fromarray(pixels).convert("RGB").resize((160, 160)), (x + 40, y + 8))
        selected = i == decision.position
        draw.text((x + 8, y + 173), f"{i + 1}. genome #{key}" + (" SELECTED" if selected else ""), fill="black")
        score = "unscored" if decision.scores is None else f"score: {decision.scores[i]:.6g}"
        draw.text((x + 8, y + 189), score, fill="black")
        if decision.evaluation is not None:
            values = decision.evaluation.values[i]
            if len(values) <= 2:
                available = decision.evaluation.metadata.get("measurement_available", [True] * len(values))
                for column, (label, value) in enumerate(zip(decision.evaluation.names, values)):
                    formatted = f"{value:.4g}" if available[column] else "unavailable"
                    draw.text((x + 8, y + 205 + 16 * column), f"{label[:24]}: {formatted}", fill="#555555")
            else:
                column = int(values.argmax())
                label = decision.evaluation.names[column]
                draw.text((x + 8, y + 205), "Largest raw measurement:", fill="#555555")
                draw.text((x + 8, y + 221), f"{label[:28]} [{column}]: {values[column]:.4g}", fill="#555555")
        if selected:
            draw.rectangle((x + 2, y + 2, x + width - 4, y + height - 4), outline="#267744", width=3)
    return sheet


def progress_message(summary, steps):
    score = summary["selected_score"]
    detail = "unscored" if score is None else f"score {score:.6g}"
    return (f"Step {summary['decisions']}/{steps}: selected #{summary['selected_id']}, "
            f"{summary['selection_mode']} ({detail}); "
            f"{summary['candidate_presentations']} candidate presentations, "
            f"{summary['evaluated_images']} image evaluations")
