import { C, base, labelBox, text, callout } from "./common.mjs";

export async function slide05(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 5, "INSERTION", "PFESA 只增强送入 decoder 的 skip");
  labelBox(slide, ctx, 100, 220, 210, 62, "Stage output", { fill: C.blueSoft, stroke: C.blue, bold: true });
  text(slide, ctx, 322, 230, 60, 46, "→", { size: 34, color: C.muted, align: "center" });
  labelBox(slide, ctx, 388, 220, 210, 62, "Optional PFESA", { fill: C.purpleSoft, stroke: C.purple, bold: true, color: C.purple });
  text(slide, ctx, 610, 230, 60, 46, "→", { size: 34, color: C.muted, align: "center" });
  labelBox(slide, ctx, 676, 220, 238, 62, "Decoder concat", { fill: C.greenSoft, stroke: C.green, bold: true, color: C.green });
  text(slide, ctx, 190, 324, 460, 52, "同时，原始 stage output 仍继续进入主干下采样：", { size: 22, color: C.ink });
  labelBox(slide, ctx, 260, 410, 210, 58, "Stage output", { fill: C.blueSoft, stroke: C.blue, bold: true });
  text(slide, ctx, 480, 418, 60, 42, "→", { size: 32, color: C.muted, align: "center" });
  labelBox(slide, ctx, 546, 410, 210, 58, "Patch Merging", { fill: C.white, stroke: C.line, bold: true });
  text(slide, ctx, 766, 418, 60, 42, "→", { size: 32, color: C.muted, align: "center" });
  labelBox(slide, ctx, 832, 410, 220, 58, "Next encoder stage", { fill: C.blueSoft, stroke: C.blue, bold: true });
  callout(slide, ctx, 92, 552, 960, 72, "关键点",
    "PFESA 不改变 merge1 / merge2 / merge3 的输入，因此不会改变 CSWin-UNet encoder 主干的逐级下采样路径。",
    C.orange, C.orangeSoft);
  return slide;
}
