import { NextResponse } from "next/server";
import { getPreset } from "@/lib/tariffs/presets";

export const dynamic = "force-static";

export async function GET(
  _req: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { id } = await context.params;
  const t = getPreset(id);
  if (!t) return new NextResponse("Not found", { status: 404 });
  return NextResponse.json(t);
}
