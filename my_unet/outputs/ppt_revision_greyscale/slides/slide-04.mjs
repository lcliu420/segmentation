import { C, base, labelBox, text, callout } from "./common.mjs";

export async function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 4, "PFESA", "FFT 高频/低频解耦后形成无参数注意力");
  labelBox(slide, ctx, 64, 260, 130, 64, "Tokens\nB x L x C", { fill: C.paper, stroke: C.line, size: 15 });
  labelBox(slide, ctx, 224, 260, 150, 64, "Reshape\nB x C x Hs x Ws", { fill: C.paper, stroke: C.line, size: 14 });
  labelBox(slide, ctx, 404, 260, 92, 64, "FFT", { fill: C.softer, stroke: C.dark, bold: true });
  labelBox(slide, ctx, 544, 214, 142, 62, "Low freq\nStructure", { fill: C.paper, stroke: C.line, size: 15 });
  labelBox(slide, ctx, 544, 338, 142, 62, "High freq\nEdge", { fill: C.softer, stroke: C.line, size: 15 });
  labelBox(slide, ctx, 744, 260, 126, 64, "SA + EA\nSigmoid", { fill: C.paper, stroke: C.dark, size: 15 });
  labelBox(slide, ctx, 924, 260, 132, 64, "Multiply\nOriginal skip", { fill: C.softer, stroke: C.line, size: 15 });
  labelBox(slide, ctx, 1084, 260, 122, 64, "Output tokens\nB x L x C", { fill: C.paper, stroke: C.line, size: 14 });
  [198, 380, 500, 884, 1060].forEach((x) => text(slide, ctx, x, 270, 24, 40, "→", { size: 24, color: C.mid, align: "center" }));
  text(slide, ctx, 704, 280, 34, 34, "+", { size: 28, bold: true, color: C.dark, align: "center" });
  callout(slide, ctx, 106, 470, 360, 92, "高频分支",
    "更偏向边界、细节和梯度敏感区域，可服务于边界增强。");
  callout(slide, ctx, 520, 470, 360, 92, "低频分支",
    "更偏向整体结构建模，有助于减少噪声或伪边界干扰。");
  callout(slide, ctx, 934, 470, 190, 92, "参数量",
    "PFESA 本身不引入额外可训练参数。");
  return slide;
}
