"use client";

import Link from "next/link";
import { Lock } from "lucide-react";
import { Card, CardBody, Button } from "@/components/ui/primitives";

/** Shown on the watchlist / paper pages when no user is signed in. */
export function SignInNotice({ feature }: { feature: string }) {
  return (
    <Card>
      <CardBody className="flex flex-col items-center gap-3 py-16 text-center">
        <span className="grid h-14 w-14 place-items-center rounded-2xl bg-brand/10 ring-1 ring-brand/20">
          <Lock className="h-6 w-6 text-brand" strokeWidth={1.75} />
        </span>
        <p className="text-sm text-muted">Sign in to use your {feature}. It syncs across your devices and stays private to you.</p>
        <Link href="/login">
          <Button>Sign in</Button>
        </Link>
      </CardBody>
    </Card>
  );
}
