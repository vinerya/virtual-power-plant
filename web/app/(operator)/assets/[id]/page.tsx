import Link from "next/link";
import { ChevronLeft } from "lucide-react";
import { AssetDetail } from "@/components/asset/asset-detail";

export default async function AssetPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return (
    <div className="space-y-4">
      <div>
        <Link
          href="/"
          className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
        >
          <ChevronLeft className="h-4 w-4" />
          Back to fleet
        </Link>
      </div>
      <AssetDetail id={id} />
    </div>
  );
}
