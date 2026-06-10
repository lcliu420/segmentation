import { C, base, text, box } from "./common.mjs";

const rows = [
  ["x123", "86", "0.885479", "+0.001014", "0.802455", "0.337396", "19.175577", "0.059714"],
  ["x23", "72", "0.885237", "+0.000772", "0.801105", "0.336616", "19.870700", "0.060038"],
  ["x13", "62", "0.884897", "+0.000432", "0.801481", "0.340406", "19.502356", "0.059972"],
  ["x2", "78", "0.884716", "+0.000250", "0.799833", "0.331567", "19.712700", "0.059763"],
  ["baseline", "61", "0.884466", "0", "0.800809", "0.338314", "19.663332", "0.059606"],
  ["x12", "98", "0.884278", "-0.000188", "0.800294", "0.337150", "20.416333", "0.060493"],
  ["x3", "63", "0.882825", "-0.001641", "0.797871", "0.333621", "20.143431", "0.061558"],
  ["x1", "72", "0.882699", "-0.001766", "0.797907", "0.334613", "20.108473", "0.062094"],
];

export async function slide08(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 8, "RESULTS", "outputs_new 验证集 best epoch 结果");
  const headers = ["模型", "epoch", "Dice", "Delta", "IoU", "B-IoU", "HD95", "MAE"];
  const widths = [116, 86, 132, 122, 132, 132, 132, 118];
  let x = 58;
  const y0 = 145;
  headers.forEach((h, i) => {
    box(slide, ctx, x, y0, widths[i], 40, { fill: C.ink, stroke: C.ink });
    text(slide, ctx, x + 6, y0 + 8, widths[i] - 12, 24, h, { size: 15, color: C.white, bold: true, align: "center" });
    x += widths[i];
  });
  rows.forEach((r, ri) => {
    x = 58;
    const y = y0 + 40 + ri * 44;
    const isBest = r[0] === "x123";
    const isBoundary = r[0] === "x13";
    r.forEach((v, ci) => {
      const fill = isBest ? C.greenSoft : isBoundary ? C.orangeSoft : ri % 2 ? "#F8FAFC" : C.white;
      box(slide, ctx, x, y, widths[ci], 44, { fill, stroke: C.line });
      const color = (isBest && (ci === 0 || ci === 2)) || (isBoundary && ci === 5) ? C.ink : C.ink;
      text(slide, ctx, x + 5, y + 11, widths[ci] - 10, 22, v, {
        size: 13.5, color, bold: isBest || (isBoundary && ci === 5), align: "center",
      });
      x += widths[ci];
    });
  });
  text(slide, ctx, 70, 570, 1040, 26, "高亮：x123 综合 Dice 最优；x13 Boundary IoU 最优；x1/x3 单独加入 PFESA 低于 baseline。", {
    size: 19, color: C.orange, bold: true,
  });
  text(slide, ctx, 70, 606, 900, 24, "说明：该页仅为 new_dataset 验证集 best epoch 结果，不能直接作为最终测试集结论。", {
    size: 16, color: C.muted,
  });
  return slide;
}
