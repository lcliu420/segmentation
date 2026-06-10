import { C, base, labelBox, text, box } from "./common.mjs";

export async function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 3, "REFERENCES", "两篇论文分别提供了什么");
  box(slide, ctx, 74, 155, 500, 420, { fill: C.blueSoft, stroke: C.blue });
  box(slide, ctx, 684, 155, 500, 420, { fill: C.purpleSoft, stroke: C.purple });
  text(slide, ctx, 104, 178, 430, 34, "CSWin-UNet 论文", { size: 26, bold: true, color: C.blue });
  text(slide, ctx, 714, 178, 430, 34, "PFESA 论文", { size: 26, bold: true, color: C.purple });
  [
    ["Convolutional Token Embedding", 104, 252],
    ["CSWin Transformer Block", 104, 318],
    ["U-shaped Encoder-Decoder", 104, 384],
    ["CARAFE Upsampling + Skip", 104, 450],
  ].forEach(([v, x, y]) => labelBox(slide, ctx, x, y, 380, 44, v, { fill: C.white, stroke: C.blue, color: C.ink }));
  [
    ["FFT Frequency Decoupling", 714, 252],
    ["High-frequency Edge Attention", 714, 318],
    ["Low-frequency Structure Attention", 714, 384],
    ["Parameter-free Skip Refinement", 714, 450],
  ].forEach(([v, x, y]) => labelBox(slide, ctx, x, y, 390, 44, v, { fill: C.white, stroke: C.purple, color: C.ink }));
  text(slide, ctx, 594, 345, 90, 45, "+", { size: 44, bold: true, color: C.orange, align: "center" });
  text(slide, ctx, 148, 606, 940, 34, "组合迁移：保留 CSWin-UNet 主网络，只把 PFESA 接到可配置 skip feature。", {
    size: 22, color: C.ink, bold: true, align: "center",
  });
  return slide;
}
