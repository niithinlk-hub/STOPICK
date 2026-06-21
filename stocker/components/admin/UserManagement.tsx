"use client";

import { useEffect, useState } from "react";
import { Loader2, ShieldCheck, Trash2, UserPlus } from "lucide-react";
import { cn } from "@/lib/cn";
import { Button, Card, CardBody, CardHeader, CardTitle } from "@/components/ui/primitives";

interface UserRow {
  id: string;
  email: string | null;
  createdAt: string;
  lastSignInAt: string | null;
  confirmed: boolean;
  isAdmin: boolean;
}

const fmtDate = (iso: string | null) => {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
};

/** Admin-only account management: list, add (email + password), delete. Hidden for non-admins. */
export function UserManagement({ currentUserId }: { currentUserId?: string | null }) {
  const [users, setUsers] = useState<UserRow[] | null>(null);
  const [forbidden, setForbidden] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ text: string; ok: boolean } | null>(null);

  const load = async () => {
    try {
      const r = await fetch("/api/users");
      if (r.status === 403) {
        setForbidden(true);
        setUsers([]);
        return;
      }
      const d = await r.json();
      setUsers(d.users ?? []);
    } catch {
      setUsers([]);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const add = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setMsg(null);
    try {
      const r = await fetch("/api/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error ?? "Failed to create user.");
      setMsg({ text: `Created ${email}.`, ok: true });
      setEmail("");
      setPassword("");
      load();
    } catch (err) {
      setMsg({ text: err instanceof Error ? err.message : "Failed.", ok: false });
    } finally {
      setBusy(false);
    }
  };

  const del = async (id: string, em: string | null) => {
    if (!window.confirm(`Delete ${em ?? "this user"}? Their account and saved data are removed permanently.`)) return;
    setBusy(true);
    setMsg(null);
    try {
      const r = await fetch(`/api/users/${id}`, { method: "DELETE" });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error ?? "Failed to delete.");
      setMsg({ text: "User deleted.", ok: true });
      load();
    } catch (err) {
      setMsg({ text: err instanceof Error ? err.message : "Failed.", ok: false });
    } finally {
      setBusy(false);
    }
  };

  if (forbidden) return null;

  const field =
    "w-full rounded-lg border border-border bg-elevated/60 px-3 py-2 text-sm text-text placeholder:text-faint focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60";
  const lbl = "text-2xs font-medium uppercase tracking-widest text-faint";

  return (
    <Card>
      <CardHeader>
        <CardTitle>User management</CardTitle>
      </CardHeader>
      <CardBody className="space-y-5">
        {/* Add user */}
        <form onSubmit={add} className="flex flex-wrap items-end gap-3">
          <div className="flex-1 space-y-1.5" style={{ minWidth: "12rem" }}>
            <label htmlFor="newuser-email" className={lbl}>
              Email
            </label>
            <input
              id="newuser-email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="person@email.com"
              autoComplete="off"
              className={field}
            />
          </div>
          <div className="flex-1 space-y-1.5" style={{ minWidth: "12rem" }}>
            <label htmlFor="newuser-pw" className={lbl}>
              Temporary password
            </label>
            <input
              id="newuser-pw"
              type="text"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="At least 6 characters"
              autoComplete="off"
              className={field}
            />
          </div>
          <Button type="submit" disabled={busy || !email.includes("@") || password.length < 6}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin" strokeWidth={2} /> : <UserPlus className="h-4 w-4" strokeWidth={2} />}
            Add user
          </Button>
        </form>

        {msg && <p className={cn("text-xs", msg.ok ? "text-bull" : "text-bear")}>{msg.text}</p>}

        {/* User list */}
        {users === null ? (
          <p className="text-sm text-muted">Loading…</p>
        ) : users.length === 0 ? (
          <p className="text-sm text-muted">No users.</p>
        ) : (
          <div className="overflow-x-auto rounded-lg border border-border">
            <table className="w-full text-sm">
              <thead className="bg-elevated text-2xs uppercase tracking-wider text-faint">
                <tr>
                  <th className="px-3 py-2 text-left font-semibold">Email</th>
                  <th className="px-3 py-2 text-left font-semibold">Added</th>
                  <th className="px-3 py-2 text-left font-semibold">Last sign-in</th>
                  <th className="px-3 py-2 text-right font-semibold">Action</th>
                </tr>
              </thead>
              <tbody>
                {users.map((u) => {
                  const isSelf = u.id === currentUserId;
                  return (
                    <tr key={u.id} className="border-t border-border">
                      <td className="px-3 py-2">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-medium text-text">{u.email ?? "—"}</span>
                          {u.isAdmin && (
                            <span className="inline-flex items-center gap-1 rounded bg-brand/10 px-1.5 py-0.5 text-2xs font-medium text-brand ring-1 ring-brand/30">
                              <ShieldCheck className="h-3 w-3" strokeWidth={2} />
                              admin
                            </span>
                          )}
                          {isSelf && <span className="text-2xs text-faint">(you)</span>}
                        </div>
                      </td>
                      <td className="px-3 py-2 text-muted tnum">{fmtDate(u.createdAt)}</td>
                      <td className="px-3 py-2 text-muted tnum">{fmtDate(u.lastSignInAt)}</td>
                      <td className="px-3 py-2 text-right">
                        <button
                          type="button"
                          onClick={() => del(u.id, u.email)}
                          disabled={busy || isSelf}
                          aria-label={`Delete ${u.email ?? "user"}`}
                          title={isSelf ? "You can't delete your own account" : "Delete user"}
                          className="inline-flex items-center justify-center rounded-md p-1.5 text-faint transition-colors hover:bg-bear/10 hover:text-bear focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/60 disabled:cursor-not-allowed disabled:opacity-30"
                        >
                          <Trash2 className="h-4 w-4" strokeWidth={2} />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        <p className="text-2xs text-faint">
          New users sign in at the login page with the email + temporary password you set (they can change it in
          Settings). Public sign-ups remain disabled.
        </p>
      </CardBody>
    </Card>
  );
}
