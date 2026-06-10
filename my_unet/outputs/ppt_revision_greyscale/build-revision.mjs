import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const artifactPath = "C:/Users/lcliu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";
const artifact = await import(pathToFileURL(artifactPath).href);
const { Presentation, PresentationFile } = artifact;

const workspace = process.env.WORKSPACE_DIR;
const out = process.env.OUT_PPTX;
const previewDir = path.join(workspace, "preview");
const layoutDir = path.join(workspace, "layout");
const slidesDir = path.join(workspace, "slides");

function pad(n) { return String(n).padStart(2, "0"); }
function line(fill = "#00000000", width = 0, style = "solid") { return { style, fill, width }; }
function frame(options) {
  const left = options.left ?? options.x ?? 0;
  const top = options.top ?? options.y ?? 0;
  const width = options.width ?? options.w;
  const height = options.height ?? options.h;
  return { left, top, width, height };
}
async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}
async function saveBlob(blob, outputPath) {
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, Buffer.from(await blob.arrayBuffer()));
}
function makeCtx(slideNumber) {
  return {
    W: 1280, H: 720, slideNumber, workspaceDir: workspace, line,
    addShape(slide, options) {
      const { geometry = "rect", fill = "#00000000", line: ln = line(), name, ...rest } = options;
      return slide.shapes.add({ geometry, name, position: frame(rest), fill, line: ln });
    },
    addText(slide, options) {
      const { text = "", fontSize = 24, color = "#171717", bold = false, typeface = "Microsoft YaHei", align = "left", valign = "top", fill = "#00000000", line: ln = line(), insets = { left: 0, right: 0, top: 0, bottom: 0 }, ...rest } = options;
      const shape = this.addShape(slide, { ...rest, fill, line: ln });
      shape.text = text;
      shape.text.fontSize = fontSize;
      shape.text.color = color;
      shape.text.bold = Boolean(bold);
      shape.text.typeface = typeface;
      shape.text.alignment = align;
      shape.text.verticalAlignment = valign;
      shape.text.insets = insets;
      return shape;
    },
    async addImage(slide, options) {
      const { path: imagePath, fit = "cover", alt = "", name, ...rest } = options;
      const image = slide.images.add({ blob: await readImageBlob(imagePath), fit, alt, name });
      image.position = frame(rest);
      return image;
    },
  };
}

const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
for (let i = 1; i <= 5; i += 1) {
  const mod = await import(`${pathToFileURL(path.join(slidesDir, `slide-${pad(i)}.mjs`)).href}?t=${Date.now()}`);
  await mod[`slide${pad(i)}`](presentation, makeCtx(i));
}

await fs.mkdir(previewDir, { recursive: true });
await fs.mkdir(layoutDir, { recursive: true });
for (let i = 0; i < presentation.slides.count; i += 1) {
  const slide = presentation.slides.getItem(i);
  await saveBlob(await presentation.export({ slide, format: "png", scale: 1 }), path.join(previewDir, `slide-${pad(i + 1)}.png`));
  await fs.writeFile(path.join(layoutDir, `slide-${pad(i + 1)}.layout.json`), await (await presentation.export({ slide, format: "layout" })).text(), "utf8");
}
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(out);
console.log(JSON.stringify({ out, slideCount: presentation.slides.count, bytes: (await fs.stat(out)).size }, null, 2));
