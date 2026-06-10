import { C, base, labelBox, text, callout } from "./common.mjs";

export async function slide10(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 10, "NEXT STEPS", "后续计划：补 test 结果，再决定主推结构");
  const items = [
    ["1", "补充 test split", "用各模型 best.pth 在 new_dataset test 上跑 test.py，确认泛化表现。"],
    ["2", "对照原始数据集", "new_dataset 是清洗后数据，不能和原始数据集结果混成同一组主实验。"],
    ["3", "继续模块探索", "如果要进一步贴合课题，可考虑显式边界分支或伪边界抑制监督。"],
  ];
  items.forEach((item, i) => {
    const y = 160 + i * 132;
    labelBox(slide, ctx, 92, y, 54, 54, item[0], { fill: C.ink, stroke: C.ink, color: C.white, size: 24, bold: true });
    text(slide, ctx, 178, y - 2, 420, 32, item[1], { size: 25, bold: true, color: C.ink });
    text(slide, ctx, 178, y + 42, 800, 38, item[2], { size: 20, color: C.muted });
  });
  callout(slide, ctx, 802, 548, 280, 78, "当前推荐汇报结论",
    "先把 x123 作为综合最优配置讨论，把 x13 作为边界指标上的观察点。",
    C.green, C.greenSoft);
  return slide;
}
