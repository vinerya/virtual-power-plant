"use client";

import { useState } from "react";
import { TariffList } from "./tariff-list";
import { TariffDetail } from "./tariff-detail";

export function TariffsView({ initialId = null }: { initialId?: string | null }) {
  const [selected, setSelected] = useState<string | null>(initialId);
  return (
    <div
      className="flex h-[calc(100vh-7rem)] overflow-hidden rounded-md border bg-card"
      data-testid="tariffs-view"
    >
      <TariffList selectedId={selected} onSelect={setSelected} />
      <TariffDetail tariffId={selected} />
    </div>
  );
}
