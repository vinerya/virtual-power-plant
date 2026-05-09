import { TariffsView } from "@/components/tariffs/tariffs-view";

export default async function TariffDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return (
    <div className="space-y-3">
      <header>
        <h2 className="text-2xl font-semibold tracking-tight">Tariffs</h2>
        <p className="text-sm text-muted-foreground">
          Deep-linked to <code className="font-mono">{id}</code>.
        </p>
      </header>
      <TariffsView initialId={id} />
    </div>
  );
}
