import fs from "node:fs";
import path from "node:path";
import Papa from "papaparse";

const APP_ROOT = process.cwd();
const REPO_ROOT = path.resolve(APP_ROOT, "..");
const LABELED_CSV = path.join(REPO_ROOT, "data", "processed", "labeled_dataset_merged_week_cap60.csv");
const PROCESSED_PUBLIC = path.join(APP_ROOT, "public", "data", "processed");
const GEO_PUBLIC = path.join(APP_ROOT, "public", "data", "geo");
const HISTORICAL_END_WEEK = "2020-W53";

const EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "neutral"];

const provinceBoxes = [
  ["新疆", 73, 35, 96, 49],
  ["西藏", 78, 27, 99, 36],
  ["青海", 89, 32, 103, 39],
  ["甘肃", 94, 34, 108, 42],
  ["内蒙古", 97, 39, 123, 50],
  ["黑龙江", 123, 44, 135, 53],
  ["吉林", 122, 41, 132, 45],
  ["辽宁", 120, 38, 126, 42],
  ["北京", 115, 39, 118, 41],
  ["天津", 117, 38, 119, 40],
  ["河北", 113, 36, 120, 42],
  ["山西", 110, 35, 114, 40],
  ["宁夏", 104, 36, 108, 39],
  ["陕西", 106, 32, 111, 38],
  ["河南", 111, 31, 116, 36],
  ["山东", 116, 34, 122, 38],
  ["江苏", 118, 31, 122, 35],
  ["安徽", 115, 29, 119, 34],
  ["湖北", 110, 29, 116, 33],
  ["四川", 99, 26, 108, 34],
  ["重庆", 106, 28, 110, 31],
  ["贵州", 104, 25, 109, 29],
  ["湖南", 110, 25, 115, 30],
  ["江西", 115, 25, 119, 30],
  ["浙江", 119, 27, 123, 31],
  ["福建", 117, 23, 121, 27],
  ["云南", 97, 21, 106, 29],
  ["广西", 106, 21, 112, 25],
  ["广东", 112, 20, 117, 25],
  ["海南", 109, 18, 112, 21],
  ["香港", 113.6, 21.8, 114.6, 22.8],
  ["澳门", 112.8, 21.6, 113.6, 22.4],
  ["台湾", 120, 21.8, 123, 25.5],
  ["上海", 121, 30.5, 123, 32]
];

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function toNumber(value) {
  const n = Number.parseFloat(String(value ?? "").trim());
  return Number.isFinite(n) ? n : 0;
}

function makeFeature([name, minX, minY, maxX, maxY]) {
  return {
    type: "Feature",
    properties: { name },
    geometry: {
      type: "Polygon",
      coordinates: [[
        [minX, minY],
        [maxX, minY],
        [maxX, maxY],
        [minX, maxY],
        [minX, minY]
      ]]
    }
  };
}

function writeGeoJson() {
  ensureDir(GEO_PUBLIC);
  const geoJson = {
    type: "FeatureCollection",
    name: "china_simplified_province_boxes",
    features: provinceBoxes.map(makeFeature)
  };
  fs.writeFileSync(
    path.join(GEO_PUBLIC, "china.json"),
    JSON.stringify(geoJson, null, 2),
    "utf8"
  );
  console.log(`[OK] public/data/geo/china.json (${geoJson.features.length} provinces)`);
}

function writePostExamples() {
  ensureDir(PROCESSED_PUBLIC);
  if (!fs.existsSync(LABELED_CSV)) {
    const fallbackPath = path.join(PROCESSED_PUBLIC, "post_examples.json");
    fs.writeFileSync(fallbackPath, JSON.stringify({ generated_at: new Date().toISOString(), provinces: {} }, null, 2), "utf8");
    console.warn(`[WARN] Missing ${LABELED_CSV}. Empty post_examples.json written.`);
    return;
  }

  const csvText = fs.readFileSync(LABELED_CSV, "utf8");
  const parsed = Papa.parse(csvText, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: false
  });

  const provinces = {};
  for (const row of parsed.data) {
    const province = String(row.province ?? "").trim();
    const content = String(row.content_clean ?? "").trim();
    const dateWeek = String(row.date_week ?? "").trim();
    if (!province || !content || !dateWeek || dateWeek > HISTORICAL_END_WEEK) continue;

    provinces[province] ??= {};
    for (const emotion of EMOTIONS) {
      provinces[province][emotion] ??= [];
      provinces[province][emotion].push({
        post_id: String(row.post_id ?? ""),
        date_week: dateWeek,
        date_month: String(row.date_month ?? ""),
        province,
        content,
        score: toNumber(row[emotion]),
        scores: Object.fromEntries(EMOTIONS.map((key) => [key, toNumber(row[key])]))
      });
    }
  }

  for (const byEmotion of Object.values(provinces)) {
    for (const emotion of EMOTIONS) {
      byEmotion[emotion] = (byEmotion[emotion] ?? [])
        .sort((a, b) => b.score - a.score)
        .slice(0, 5);
    }
  }

  const out = {
    generated_at: new Date().toISOString(),
    source: "data/processed/labeled_dataset_merged_week_cap60.csv",
    historical_end_week: HISTORICAL_END_WEEK,
    emotions: EMOTIONS,
    provinces
  };
  fs.writeFileSync(path.join(PROCESSED_PUBLIC, "post_examples.json"), JSON.stringify(out, null, 2), "utf8");
  console.log(`[OK] public/data/processed/post_examples.json (${Object.keys(provinces).length} provinces)`);
}

writeGeoJson();
writePostExamples();
