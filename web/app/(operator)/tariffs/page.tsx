import { TariffsView } from "@/components/tariffs/tariffs-view";

export const metadata = {
  title: "Tariffs · VPP Console",
};

export default function TariffsPage() {
  return (
    <div className="space-y-3">
      <header>
        <h2 className="text-2xl font-semibold tracking-tight">Tariffs</h2>
        <p className="text-sm text-muted-foreground">
          Browse utility tariffs, inspect TOU schedules, and simulate bills
          against synthetic or uploaded meter traces.
        </p>
      </header>
      <TariffsView />
    </div>
  );
}
