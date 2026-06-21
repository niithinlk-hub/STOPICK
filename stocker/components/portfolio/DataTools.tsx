"use client";

import { useRef, useState } from "react";
import { Download, Loader2, Upload } from "lucide-react";
import { Button } from "@/components/ui/primitives";
import { exportData, importData } from "@/lib/client/store";

/** Export / import the signed-in user's watchlist + blotter as a JSON backup file. */
export function DataTools() {
  const fileRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);

  const doExport = async () => {
    setBusy(true);
    try {
      const blob = new Blob([await exportData()], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `stocker-backup-${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setBusy(false);
    }
  };

  const doImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async () => {
      setBusy(true);
      try {
        await importData(String(reader.result));
        window.location.reload();
      } catch {
        window.alert("Could not import that file — expected a STOCKER JSON backup.");
      } finally {
        setBusy(false);
      }
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  return (
    <div className="flex gap-2">
      <Button variant="ghost" onClick={doExport} disabled={busy} aria-label="Export backup">
        {busy ? <Loader2 className="h-4 w-4 animate-spin" strokeWidth={2} /> : <Download className="h-4 w-4" strokeWidth={2} />}
        Export
      </Button>
      <Button variant="ghost" onClick={() => fileRef.current?.click()} disabled={busy} aria-label="Import backup">
        <Upload className="h-4 w-4" strokeWidth={2} />
        Import
      </Button>
      <input ref={fileRef} type="file" accept="application/json" className="hidden" onChange={doImport} />
    </div>
  );
}
