# Presentation Skill

## Intent

Generate a downloadable PowerPoint deck from uploaded materials. The skill converts
material into a small structured slide plan, then the Python execution layer builds a
real `.pptx` file with a dependency-free OpenXML writer.

## Operating rules

- Use only the provided material.
- Keep slide titles short and specific.
- Use 2-5 bullets per slide.
- Do not invent metrics, conclusions, or claims.
- If evidence is insufficient, use “材料未提及”.
- Favor interview / project-demo friendly structure: background → approach → progress → result → next step.

## Execution layer

The Python class `PresentationSkill` asks the LLM for slide JSON, validates/falls back,
then calls `pptx_builder.build_pptx()` and returns a base64 download artifact.
