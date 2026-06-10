import { C, base, labelBox, text, callout } from "./common.mjs";

export async function slide07(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, 7, "ABLATION", "消融设置：x1/x2/x3 三条 skip 的组合");
  const modes = [
    ["none", 104, 190, C.faint, C.line],
    ["x1", 300, 190, C.purpleSoft, C.purple],
    ["x2", 496, 190, C.purpleSoft, C.purple],
    ["x3", 692, 190, C.purpleSoft, C.purple],
    ["x12", 300, 306, C.blueSoft, C.blue],
    ["x13", 496, 306, C.blueSoft, C.blue],
    ["x23", 692, 306, C.blueSoft, C.blue],
    ["x123", 496, 422, C.greenSoft, C.green],
  ];
  modes.forEach(([m, x, y, fill, stroke]) => labelBox(slide, ctx, x, y, 132, 60, m, { fill, stroke, size: 24, bold: true }));
  callout(slide, ctx, 900, 190, 250, 108, "数据设置",
    "清洗后的 new_dataset\nmodal = ALL\nnum_classes = 2",
    C.blue, C.blueSoft);
  callout(slide, ctx, 900, 332, 250, 108, "指标口径",
    "当前表格来自验证集 best epoch，不是最终 test 结果。",
    C.orange, C.orangeSoft);
  text(slide, ctx, 104, 560, 760, 40, "讲解顺序建议：先解释单点 x1/x2/x3，再解释组合 x12/x13/x23/x123。", {
    size: 21, color: C.ink, bold: true,
  });
  return slide;
}
