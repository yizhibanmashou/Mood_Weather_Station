import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";

interface EChartProps {
  option: EChartsOption;
  height?: number | string;
}

/** Merge default animation config into every ECharts option */
function withAnimation(option: EChartsOption): EChartsOption {
  return {
    ...option,
    animationDuration: 900,
    animationDurationUpdate: 700,
    animationEasing: "cubicOut",
    animationEasingUpdate: "cubicOut",
  };
}

export function EChart({ option, height = 360 }: EChartProps) {
  return (
    <ReactECharts
      option={withAnimation(option)}
      notMerge
      lazyUpdate
      style={{ width: "100%", height }}
      opts={{ renderer: "canvas" }}
    />
  );
}
