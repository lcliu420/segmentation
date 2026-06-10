import { C, base, labelBox, text, box } from "./common.mjs";

export async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 2, "REFERENCES", "两篇论文分别提供了什么");
  box(slide, ctx, 74, 155, 500, 420, { fill: C.softer, stroke: C.line });
  box(slide, ctx, 684, 155, 500, 420, { fill: C.paper, stroke: C.line });
  text(slide, ctx, 104, 178, 430, 34, "CSWin-UNet 论文", { size: 26, bold: true, color: C.ink });
  text(slide, ctx, 714, 178, 430, 34, "PFESA 论文", { size: 26, bold: true, color: C.ink });
  [
    ["Convolutional Token Embedding", 104, 252],
    ["CSWin Transformer Block", 104, 318],
    ["U-shaped Encoder-Decoder", 104, 384],
    ["CARAFE Upsampling + Skip", 104, 450],
  ].forEach(([v, x, y]) => labelBox(slide, ctx, x, y, 380, 44, v, { fill: C.paper, stroke: C.line, color: C.ink }));
  [
    ["FFT Frequency Decoupling", 714, 252],
    ["High-frequency Edge Attention", 714, 318],
    ["Low-frequency Structure Attention", 714, 384],
    ["Parameter-free Skip Refinement", 714, 450],
  ].forEach(([v, x, y]) => labelBox(slide, ctx, x, y, 390, 44, v, { fill: C.softer, stroke: C.line, color: C.ink }));
  text(slide, ctx, 594, 345, 90, 45, "+", { size: 44, bold: true, color: C.dark, align: "center" });
  text(slide, ctx, 148, 606, 940, 34, "组合迁移：保留 CSWin-UNet 主网络，只把 PFESA 接到可配置 skip feature。", {
    size: 22, color: C.ink, bold: true, align: "center",
  });
  return slide;
}
