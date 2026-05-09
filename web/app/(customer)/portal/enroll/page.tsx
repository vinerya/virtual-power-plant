import { EnrollmentForm } from "@/components/customer/enrollment-form";

export const metadata = { title: "Programs · VPP Member Portal" };

export default function EnrollPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">
          Demand-response programs
        </h2>
        <p className="text-sm text-muted-foreground">
          Pick the programs you want to participate in. You can change this any
          time.
        </p>
      </div>
      <EnrollmentForm />
    </div>
  );
}
