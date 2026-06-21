/**
 * Tree-shaken ECharts setup — register only the chart types and components we use.
 * Reduces bundle size from ~800KB to ~300KB.
 */
import { BarChart, LineChart, PieChart, RadarChart, ScatterChart } from "echarts/charts";
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  DatasetComponent,
  DataZoomComponent,
  VisualMapComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([
  BarChart, LineChart, PieChart, RadarChart, ScatterChart,
  TitleComponent, TooltipComponent, LegendComponent,
  GridComponent, DatasetComponent, DataZoomComponent, VisualMapComponent,
  CanvasRenderer,
]);

export { echarts };
