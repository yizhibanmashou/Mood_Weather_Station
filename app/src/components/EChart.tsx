import ReactEChartsCore from "echarts-for-react/lib/core";
import type { EChartsOption } from "echarts";
import { echarts } from "../utils/echartsSetup";

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
    <ReactEChartsCore
      echarts={echarts}
      option={withAnimation(option)}
      notMerge
      lazyUpdate
      style={{ width: "100%", height }}
      opts={{ renderer: "canvas" }}
    />
  );
}
