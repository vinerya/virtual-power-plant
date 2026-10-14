"use client";

/**
 * Protocol messages panel for the Protocols page.
 *
 * Three views of the grid-protocol traffic the API already exposes:
 * - Events: the demand-response events the OpenADR VEN holds, with their
 *   signals, window, status and our opt-in/out answer (operators can
 *   override it; the answer is sent to the VTN when the adapter is live).
 * - Decision log: `dr_event_responses`, the persisted record of what the DR
 *   orchestrator received, answered, dispatched and released.
 * - Live feed: the `grid_events` WebSocket channel (protocol status, DR
 *   events/responses, V2G dispatch), newest first, kept in memory only.
 */

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { ChevronDown, ChevronRight, Radio } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ErrorState } from "@/components/ui/error-state";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { apiErrorMessage } from "@/lib/api/errors";
import {
  getDRStatus,
  listDRResponses,
  listOpenADREvents,
  overrideOpenADROpt,
  type DRResponse,
  type OpenADREvent,
  type OpenADREvents,
} from "@/lib/api/protocols";
import { asBusEvent, getWsClient, type BusEvent } from "@/lib/ws/client";

export const EVENTS_KEY = ["openadr-events"] as const;
export const RESPONSES_KEY = ["dr-responses"] as const;
export const DR_STATUS_KEY = ["dr-status"] as const;

const LIVE_FEED_LIMIT = 100;
const FINAL_STATUSES = new Set(["completed", "cancelled"]);

function fmtTs(ts: number | null | undefined): string {
  if (ts === null || ts === undefined || !Number.isFinite(ts)) return "—";
  return new Date(ts * 1000).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function fmtDuration(s: number): string {
  if (!Number.isFinite(s) || s <= 0) return "open-ended";
  const h = Math.floor(s / 3600);
  const m = Math.round((s % 3600) / 60);
  if (h && m) return `${h}h ${m}m`;
  if (h) return `${h}h`;
  return `${m}m`;
}

function fmtKw(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return `${v.toFixed(1)} kW`;
}

function shortId(id: string): string {
  return id.length > 12 ? `${id.slice(0, 8)}…` : id;
}

function StatusBadge({ status }: { status: string }) {
  const variant =
    status === "active"
      ? "success"
      : status === "cancelled"
        ? "destructive"
        : status === "completed"
          ? "secondary"
          : "outline";
  return <Badge variant={variant}>{status}</Badge>;
}

function OptBadge({ opt }: { opt: string | null | undefined }) {
  if (!opt) return <span className="text-muted-foreground">pending</span>;
  return <Badge variant={opt === "optIn" ? "success" : "destructive"}>{opt}</Badge>;
}

function ActionBadge({ action }: { action: string }) {
  const variant =
    action === "dispatched"
      ? "success"
      : action === "failed"
        ? "destructive"
        : action === "released" || action === "noop"
          ? "secondary"
          : "outline";
  return (
    <Badge variant={variant} className="font-mono text-[11px]">
      {action}
    </Badge>
  );
}

export function ProtocolMessages({ canOperate }: { canOperate: boolean }) {
  const qc = useQueryClient();
  const events = useQuery({
    queryKey: EVENTS_KEY,
    queryFn: listOpenADREvents,
    refetchInterval: 5_000,
  });
  const status = useQuery({
    queryKey: DR_STATUS_KEY,
    queryFn: getDRStatus,
    refetchInterval: 10_000,
  });

  // Live feed from the grid_events channel; DR events also refresh the tables.
  const [feed, setFeed] = useState<BusEvent[]>([]);
  useEffect(() => {
    const client = getWsClient();
    client.start({ channels: ["grid_events"] });
    const off = client.on((msg) => {
      if (msg.channel !== "grid_events") return;
      const ev = asBusEvent(msg.data);
      if (!ev) return;
      setFeed((prev) => [ev, ...prev].slice(0, LIVE_FEED_LIMIT));
      const t = ev.event_type.toLowerCase();
      if (t.startsWith("dr_") || t.includes("dispatch")) {
        qc.invalidateQueries({ queryKey: EVENTS_KEY });
        qc.invalidateQueries({ queryKey: RESPONSES_KEY });
        qc.invalidateQueries({ queryKey: DR_STATUS_KEY });
      }
    });
    return () => {
      off();
      client.unsubscribe("grid_events");
    };
  }, [qc]);

  const header = events.data;
  return (
    <Card data-testid="protocol-messages">
      <CardHeader className="space-y-2 pb-2">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <CardTitle className="text-base text-foreground">Protocol messages</CardTitle>
            <p className="text-xs text-muted-foreground">
              Demand-response events from the grid operator, what the VPP answered and did, and
              the live event stream.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-1.5 text-xs" data-testid="dr-summary">
            {header && (
              <>
                <Badge variant={header.registered ? "success" : "outline"}>
                  {header.registered ? "registered" : "not registered"}
                </Badge>
                {header.vtn_id && (
                  <span className="text-muted-foreground">
                    VTN <span className="font-mono text-foreground">{header.vtn_id}</span>
                  </span>
                )}
                {header.ven_id && (
                  <span className="text-muted-foreground">
                    VEN <span className="font-mono text-foreground">{header.ven_id}</span>
                  </span>
                )}
              </>
            )}
            {status.data && (
              <Badge variant={status.data.auto_response_enabled ? "success" : "secondary"}>
                auto-response {status.data.auto_response_enabled ? "on" : "off"}
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="events">
          <TabsList>
            <TabsTrigger value="events">
              Events{header ? ` (${header.events.length})` : ""}
            </TabsTrigger>
            <TabsTrigger value="log">Decision log</TabsTrigger>
            <TabsTrigger value="feed">
              <span className="inline-flex items-center gap-1">
                <Radio className="h-3 w-3" aria-hidden="true" />
                Live feed{feed.length ? ` (${feed.length})` : ""}
              </span>
            </TabsTrigger>
          </TabsList>
          <TabsContent value="events" className="pt-3">
            <EventsTable
              data={events.data}
              isLoading={events.isLoading}
              error={events.error}
              onRetry={() => void events.refetch()}
              canOperate={canOperate}
            />
          </TabsContent>
          <TabsContent value="log" className="pt-3">
            <DecisionLog />
          </TabsContent>
          <TabsContent value="feed" className="pt-3">
            <LiveFeed feed={feed} />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

function EventsTable({
  data,
  isLoading,
  error,
  onRetry,
  canOperate,
}: {
  data: OpenADREvents | undefined;
  isLoading: boolean;
  error: unknown;
  onRetry: () => void;
  canOperate: boolean;
}) {
  if (isLoading) return <Skeleton className="h-24 w-full" />;
  if (error || !data)
    return (
      <ErrorState
        title="Failed to load demand-response events."
        error={error}
        onRetry={onRetry}
      />
    );
  if (data.events.length === 0)
    return (
      <p className="text-sm text-muted-foreground" data-testid="dr-events-empty">
        No events. The VEN is {data.registered ? "registered and polling" : "not registered"}
        {data.simulated ? " (simulated: nothing is exchanged with a VTN)" : ""}; events appear
        here when the VTN distributes them.
      </p>
    );
  const rows = [...data.events].sort((a, b) => b.start_time - a.start_time);
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm" data-testid="dr-events-table">
        <thead className="text-left text-xs text-muted-foreground">
          <tr>
            <th className="py-2 pr-3 font-medium">Event</th>
            <th className="py-2 pr-3 font-medium">Signal</th>
            <th className="py-2 pr-3 font-medium">Window</th>
            <th className="py-2 pr-3 font-medium">Status</th>
            <th className="py-2 pr-3 font-medium">Our answer</th>
            {canOperate && <th className="py-2 font-medium">Override</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((e) => (
            <EventRow
              key={`${e.event_id}:${e.modification_number ?? 0}`}
              e={e}
              canOperate={canOperate}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EventRow({ e, canOperate }: { e: OpenADREvent; canOperate: boolean }) {
  const qc = useQueryClient();
  const [open, setOpen] = useState(false);
  const override = useMutation({
    mutationFn: (opt: "optIn" | "optOut") =>
      overrideOpenADROpt(e.event_id, opt, "operator override"),
    onSuccess: (r) => {
      toast.success(`Event ${shortId(e.event_id)}: ${r.opt_type}`, {
        description: r.sent_to_vtn ? "Sent to the VTN." : "Recorded locally (adapter not live).",
      });
    },
    onError: (err: unknown) => {
      toast.error("Opt override failed", {
        description: apiErrorMessage(err, "The VTN did not accept the response."),
      });
    },
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: EVENTS_KEY });
      void qc.invalidateQueries({ queryKey: RESPONSES_KEY });
    },
  });
  const final = FINAL_STATUSES.has(e.status);
  const colSpan = canOperate ? 6 : 5;
  const primary = e.signals[0];
  return (
    <>
      <tr className="border-t align-top" data-testid={`dr-event-${e.event_id}`}>
        <td className="py-2 pr-3">
          <button
            type="button"
            className="inline-flex items-center gap-1 font-mono text-xs"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            aria-label={`${open ? "Hide" : "Show"} signals for event ${e.event_id}`}
          >
            {open ? (
              <ChevronDown className="h-3 w-3" aria-hidden="true" />
            ) : (
              <ChevronRight className="h-3 w-3" aria-hidden="true" />
            )}
            {shortId(e.event_id)}
          </button>
          <div className="text-[11px] text-muted-foreground">
            {e.source === "vtn" ? "from VTN" : (e.source ?? "")}
            {e.vtn_status ? ` · ${e.vtn_status}` : ""}
            {e.test_event ? " · test" : ""}
            {e.modification_number ? ` · rev ${e.modification_number}` : ""}
          </div>
        </td>
        <td className="py-2 pr-3">
          <div className="font-medium">{e.signal_type}</div>
          <div className="text-xs text-muted-foreground">
            level {e.signal_level}
            {primary ? ` · ${primary.signal_name}/${primary.signal_type}` : ""}
          </div>
        </td>
        <td className="py-2 pr-3 whitespace-nowrap text-xs">
          <div>{fmtTs(e.start_time)}</div>
          <div className="text-muted-foreground">{fmtDuration(e.duration_seconds)}</div>
        </td>
        <td className="py-2 pr-3">
          <StatusBadge status={e.status} />
        </td>
        <td className="py-2 pr-3">
          <OptBadge opt={e.opt_type} />
          {e.opted_at ? (
            <div className="text-[11px] text-muted-foreground">{fmtTs(e.opted_at)}</div>
          ) : null}
        </td>
        {canOperate && (
          <td className="py-2">
            <div className="flex gap-1">
              <Button
                type="button"
                size="sm"
                variant="outline"
                disabled={final || override.isPending || e.opt_type === "optIn"}
                onClick={() => override.mutate("optIn")}
                aria-label={`Opt in to event ${e.event_id}`}
              >
                Opt in
              </Button>
              <Button
                type="button"
                size="sm"
                variant="outline"
                disabled={final || override.isPending || e.opt_type === "optOut"}
                onClick={() => override.mutate("optOut")}
                aria-label={`Opt out of event ${e.event_id}`}
              >
                Opt out
              </Button>
            </div>
          </td>
        )}
      </tr>
      {open && (
        <tr className="bg-muted/40" data-testid={`dr-event-${e.event_id}-signals`}>
          <td colSpan={colSpan} className="px-3 py-2 text-xs">
            <dl className="mb-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-muted-foreground">
              <dt>Event id</dt>
              <dd className="break-all font-mono text-foreground">{e.event_id}</dd>
              <dt>Market context</dt>
              <dd className="font-mono text-foreground">{e.market_context ?? "—"}</dd>
              <dt>Response required</dt>
              <dd className="text-foreground">{e.response_required ?? "—"}</dd>
              <dt>Priority</dt>
              <dd className="text-foreground">{e.priority ?? 0}</dd>
            </dl>
            {e.signals.length === 0 ? (
              <p className="text-muted-foreground">No signals.</p>
            ) : (
              e.signals.map((s, i) => (
                <div key={s.signal_id ?? i} className="mb-1">
                  <span className="font-medium">
                    {s.signal_name}/{s.signal_type}
                  </span>
                  <span className="text-muted-foreground"> · current {s.current_value ?? "—"}</span>
                  <ul className="ml-4 list-disc">
                    {s.intervals.map((iv, j) => (
                      <li key={iv.uid ?? j}>
                        {fmtTs(iv.start_time)} for {fmtDuration(iv.duration_seconds)}: value{" "}
                        <span className="font-mono">{iv.value ?? "—"}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))
            )}
          </td>
        </tr>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Decision log
// ---------------------------------------------------------------------------

function DecisionLog() {
  const q = useQuery({
    queryKey: RESPONSES_KEY,
    queryFn: () => listDRResponses(50),
    refetchInterval: 10_000,
  });
  if (q.isLoading) return <Skeleton className="h-24 w-full" />;
  if (q.error || !q.data)
    return (
      <ErrorState
        title="Failed to load the demand-response log."
        error={q.error}
        onRetry={() => void q.refetch()}
      />
    );
  if (q.data.length === 0)
    return (
      <p className="text-sm text-muted-foreground" data-testid="dr-log-empty">
        Nothing recorded yet. Every received event, opt decision, dispatch and release is logged
        here.
      </p>
    );
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm" data-testid="dr-log-table">
        <thead className="text-left text-xs text-muted-foreground">
          <tr>
            <th className="py-2 pr-3 font-medium">When</th>
            <th className="py-2 pr-3 font-medium">Action</th>
            <th className="py-2 pr-3 font-medium">Event</th>
            <th className="py-2 pr-3 font-medium">Signal</th>
            <th className="py-2 pr-3 font-medium">Answer</th>
            <th className="py-2 pr-3 text-right font-medium">Target</th>
            <th className="py-2 pr-3 text-right font-medium">Delivered</th>
            <th className="py-2 font-medium">Reason</th>
          </tr>
        </thead>
        <tbody>
          {q.data.map((r: DRResponse) => (
            <tr key={String(r.id)} className="border-t align-top" data-testid={`dr-log-${r.id}`}>
              <td className="py-2 pr-3 whitespace-nowrap text-xs text-muted-foreground">
                {fmtTs(r.created_at)}
              </td>
              <td className="py-2 pr-3">
                <ActionBadge action={r.action} />
              </td>
              <td className="py-2 pr-3 font-mono text-xs">
                <span className="text-muted-foreground">{r.protocol}</span> {shortId(r.source_id)}
                {r.revision ? <span className="text-muted-foreground"> r{r.revision}</span> : null}
              </td>
              <td className="py-2 pr-3 text-xs">
                {r.signal_type ?? "—"}
                {r.signal_level !== null && r.signal_level !== undefined
                  ? ` · ${r.signal_level}`
                  : ""}
              </td>
              <td className="py-2 pr-3">
                <OptBadge opt={r.opt_type} />
              </td>
              <td className="py-2 pr-3 text-right tabular-nums">{fmtKw(r.target_kw)}</td>
              <td className="py-2 pr-3 text-right tabular-nums">{fmtKw(r.delivered_kw)}</td>
              <td className="py-2 break-words text-xs text-muted-foreground">{r.reason || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Live feed
// ---------------------------------------------------------------------------

const SUMMARY_KEYS = ["protocol", "event_id", "opt_type", "decision", "target_kw", "status", "reason"];

function summarize(ev: BusEvent): string {
  const d = ev.data;
  const parts: string[] = [];
  for (const k of SUMMARY_KEYS) {
    const v = d[k];
    if (v === undefined || v === null || v === "") continue;
    parts.push(`${k}=${typeof v === "object" ? JSON.stringify(v) : String(v)}`);
  }
  if (parts.length === 0) {
    const s = JSON.stringify(d);
    return s.length > 160 ? `${s.slice(0, 157)}…` : s;
  }
  return parts.join("  ");
}

function tsOf(ev: BusEvent): number {
  return typeof ev.timestamp === "number" ? ev.timestamp : Date.parse(String(ev.timestamp)) / 1000;
}

function LiveFeed({ feed }: { feed: BusEvent[] }) {
  return (
    <div className="space-y-2" data-testid="live-feed">
      <p className="text-xs text-muted-foreground">
        <code>grid_events</code> WebSocket channel, newest first. The last {LIVE_FEED_LIMIT} are
        kept in this tab only; nothing here is persisted.
      </p>
      {feed.length === 0 ? (
        <p className="text-sm text-muted-foreground" data-testid="live-feed-empty">
          Waiting for events…
        </p>
      ) : (
        <ol className="divide-y text-xs" aria-live="polite">
          {feed.map((ev) => (
            <li key={ev.event_id} className="grid grid-cols-[9rem_11rem_1fr] gap-2 py-1.5">
              <span className="whitespace-nowrap text-muted-foreground">{fmtTs(tsOf(ev))}</span>
              <span className="font-mono">{ev.event_type}</span>
              <span className="break-all font-mono text-muted-foreground">{summarize(ev)}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
