import { NextResponse } from "next/server";
import { listPresetSummaries } from "@/lib/tariffs/presets";

export const dynamic = "force-static";

export function GET() {
  return NextResponse.json(listPresetSummaries());
}
