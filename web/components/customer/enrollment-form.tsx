"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { useMutation, useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { listPrograms, postEnrollment } from "@/lib/api/customer";
import type { DRProgram } from "@/lib/api/types";

const Schema = z.object({
  program_ids: z.array(z.string()).min(1, "Pick at least one program"),
  acknowledged: z.literal(true, {
    errorMap: () => ({ message: "You must accept the terms" }),
  }),
});

interface FormValues {
  program_ids: string[];
  acknowledged: boolean;
}

export function EnrollmentForm() {
  const programs = useQuery({
    queryKey: ["customer", "programs"],
    queryFn: listPrograms,
  });

  const [submitted, setSubmitted] = useState<string[] | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
    setValue,
    watch,
  } = useForm<FormValues>({
    defaultValues: { program_ids: [], acknowledged: false },
  });

  const selected = watch("program_ids") ?? [];
  const ack = watch("acknowledged");

  const m = useMutation({
    mutationFn: (v: FormValues) =>
      postEnrollment({
        program_ids: v.program_ids,
        acknowledged: v.acknowledged,
      }),
    onSuccess: (resp) => {
      toast.success("Enrolled");
      setSubmitted(resp.enrolled);
    },
    onError: () => toast.error("Enrollment failed"),
  });

  const toggle = (id: string) => {
    const next = selected.includes(id)
      ? selected.filter((x) => x !== id)
      : [...selected, id];
    setValue("program_ids", next, { shouldValidate: true });
  };

  return (
    <form
      noValidate
      onSubmit={handleSubmit((values) => {
        const parsed = Schema.safeParse(values);
        if (!parsed.success) {
          toast.error(parsed.error.errors[0]?.message ?? "Invalid form");
          return;
        }
        m.mutate(parsed.data);
      })}
      className="space-y-4"
      data-testid="enrollment-form"
    >
      {programs.isLoading ? (
        <Skeleton className="h-48 w-full" />
      ) : programs.isError || !programs.data ? (
        <p className="rounded-md border border-dashed p-4 text-sm text-destructive">
          Failed to load programs.
        </p>
      ) : (
        <div className="grid gap-3 md:grid-cols-2">
          {programs.data.map((p) => (
            <ProgramCard
              key={p.id}
              program={p}
              checked={selected.includes(p.id)}
              onToggle={() => toggle(p.id)}
            />
          ))}
        </div>
      )}

      {errors.program_ids && (
        <p className="text-xs text-destructive">{errors.program_ids.message}</p>
      )}

      <Card>
        <CardHeader className="pb-2">
          <CardTitle>Terms</CardTitle>
        </CardHeader>
        <CardContent>
          <label className="flex items-start gap-2 text-sm">
            <input
              type="checkbox"
              {...register("acknowledged")}
              className="mt-0.5"
              data-testid="ack-checkbox"
            />
            <span>
              I authorize my utility and the VPP operator to dispatch enrolled
              devices during program events. Incentives and event timings are
              described per program above.
            </span>
          </label>
          {errors.acknowledged && (
            <p className="mt-1 text-xs text-destructive">
              {errors.acknowledged.message as string}
            </p>
          )}
        </CardContent>
      </Card>

      <div className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground">
          Selected: {selected.length} program{selected.length === 1 ? "" : "s"}
        </p>
        <Button
          type="submit"
          disabled={m.isPending || selected.length === 0 || !ack}
          data-testid="enroll-submit"
        >
          {m.isPending ? "Enrolling…" : "Enroll"}
        </Button>
      </div>

      {submitted && (
        <p
          className="rounded-md border border-emerald-500/40 bg-emerald-50 p-3 text-sm text-emerald-700"
          data-testid="enrollment-success"
        >
          You are enrolled in {submitted.length} program
          {submitted.length === 1 ? "" : "s"}. Welcome aboard!
        </p>
      )}
    </form>
  );
}

function ProgramCard({
  program,
  checked,
  onToggle,
}: {
  program: DRProgram;
  checked: boolean;
  onToggle: () => void;
}) {
  return (
    <label
      className={`flex cursor-pointer flex-col gap-1.5 rounded-md border p-3 text-sm transition-colors hover:bg-muted/40 ${
        checked ? "border-primary bg-primary/5" : ""
      }`}
      data-testid="program-card"
      data-program-id={program.id}
    >
      <span className="flex items-center justify-between">
        <span className="font-medium">{program.name}</span>
        <input
          type="checkbox"
          checked={checked}
          onChange={onToggle}
          aria-label={`Enroll ${program.name}`}
        />
      </span>
      <span className="text-xs text-muted-foreground">{program.description}</span>
      {program.incentive_per_event != null && (
        <span className="text-xs font-medium text-emerald-600">
          ~${program.incentive_per_event.toFixed(0)} per event
        </span>
      )}
    </label>
  );
}
