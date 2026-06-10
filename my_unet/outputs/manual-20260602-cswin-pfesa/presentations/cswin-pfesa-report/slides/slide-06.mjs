import { C, base, labelBox, text, callout } from "./common.mjs";

export async function slide06(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 6, "PFESA", "FFT 高频/低频解耦后形成无参数注意力");
  const steps = [
    ["Tokens\nB x L x C", 64, C.white, C.line],
    ["Reshape\nB x C x Hs x Ws", 224, C.white, C.line],
    ["FFT", 404, C.orangeSoft, C.orange],
    ["Low freq\nStructure", 544, C.blueSoft, C.blue],
    ["High freq\nEdge", 544, C.purpleSoft, C.purple],
    ["SA + EA\nSigmoid", 744, C.orangeSoft, C.orange],
    ["Multiply\nOriginal skip", 924, C.greenSoft, C.green],
    ["Output tokens\nB x L x C", 1084, C.white, C.line],
  ];
  labelBox(slide, ctx, steps[0][1], 260, 130, 64, steps[0][0], { fill: steps[0][2], stroke: steps[0][3], size: 16 });
  labelBox(slide, ctx, steps[1][1], 260, 150, 64, steps[1][0], { fill: steps[1][2], stroke: steps[1][3], size: 15 });
  labelBox(slide, ctx, steps[2][1], 260, 92, 64, steps[2][0], { fill: steps[2][2], stroke: steps[2][3], bold: true });
  labelBox(slide, ctx, steps[3][1], 214, 142, 62, steps[3][0], { fill: steps[3][2], stroke: steps[3][3], size: 15 });
  labelBox(slide, ctx, steps[4][1], 338, 142, 62, steps[4][0], { fill: steps[4][2], stroke: steps[4][3], size: 15 });
  labelBox(slide, ctx, steps[5][1], 260, 126, 64, steps[5][0], { fill: steps[5][2], stroke: steps[5][3], size: 15 });
  labelBox(slide, ctx, steps[6][1], 260, 132, 64, steps[6][0], { fill: steps[6][2], stroke: steps[6][3], size: 15 });
  labelBox(slide, ctx, steps[7][1], 260, 122, 64, steps[7][0], { fill: steps[7][2], stroke: steps[7][3], size: 15 });
  [198, 380, 500, 884, 1060].forEach((x) => text(slide, ctx, x, 270, 24, 40, "→", { size: 24, color: C.muted, align: "center" }));
  text(slide, ctx, 704, 280, 34, 34, "+", { size: 28, bold: true, color: C.orange, align: "center" });
  callout(slide, ctx, 106, 470, 360, 92, "高频分支",
    "更偏向边界、细节和梯度敏感区域，可服务于边界增强。",
    C.purple, C.purpleSoft);
  callout(slide, ctx, 520, 470, 360, 92, "低频分支",
    "更偏向整体结构建模，有助于减少噪声或伪边界干扰。",
    C.blue, C.blueSoft);
  callout(slide, ctx, 934, 470, 190, 92, "参数量",
    "PFESA 本身不引入额外可训练参数。",
    C.green, C.greenSoft);
  return slide;
}
