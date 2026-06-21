import { Suspense } from "react";
import { ScannerClient } from "@/components/scanner/ScannerClient";

export default function ScannerPage() {
  return (
    <Suspense fallback={null}>
      <ScannerClient />
    </Suspense>
  );
}
