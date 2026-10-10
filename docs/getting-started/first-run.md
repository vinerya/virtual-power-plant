# First run

The API needs a database schema, a signing key and an admin account.
There is no self-registration: the first admin is created from the command
line or bootstrapped from a password file.

## With Docker Compose

The API container has already migrated the database. Create the admin (you
are prompted for the password twice; weak passwords are refused):

```bash
docker compose exec vpp-api vpp users create-admin admin
```

Then open the console at <http://localhost:3000> and the interactive API
docs at <http://localhost:8000/docs>. To create the admin on first boot
instead, see [bootstrap the first admin](../deployment.md#create-the-first-admin).

## Without Docker

From the directory you want the SQLite database (`./vpp.db`) in, with the
package installed as in [Installation](installation.md):

```bash
export VPP_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(48))")
vpp migrate
```

`vpp migrate` applies the alembic migrations shipped inside the package
(`VPP_DATABASE_URL` selects the database; SQLite `./vpp.db` by default,
`postgresql+asyncpg://...` for PostgreSQL with the `migrations` extra).

### Create the first admin

Either with the CLI (reads `--password-stdin`, `--password-file`,
`$VPP_ADMIN_PASSWORD` or an interactive prompt, in that order):

```bash
export VPP_ADMIN_PASSWORD='pick-a-long-passphrase'
vpp users create-admin admin
```

or let the API create it at startup while the `users` table is empty. The
password is read from a file, never from the environment, and startup
fails if it is missing or too weak:

```bash
printf '%s' 'pick-a-long-passphrase' > admin_password
export VPP_BOOTSTRAP_ADMIN_USERNAME=admin
export VPP_BOOTSTRAP_ADMIN_PASSWORD_FILE=$PWD/admin_password
```

Once any user exists both settings are ignored.

### Start the API

```bash
vpp serve
```

It listens on <http://127.0.0.1:8000> (`--host`, `--port`, `--workers` or
the `VPP_API_*` settings change that). Check it and sign in:

```bash
curl -s localhost:8000/health
TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/token \
  -d username=admin -d password="$VPP_ADMIN_PASSWORD" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
curl -s localhost:8000/api/v1/auth/me -H "Authorization: Bearer $TOKEN"
```

Interactive API docs are at <http://localhost:8000/docs>; the same schema
is rendered on this site under [REST API](../reference/rest-api.md).

### Start the web console

The console is a Next.js app in `web/` of the repository (Node.js 20+ and
pnpm 9, e.g. via `corepack enable`):

```bash
cd web
cp .env.example .env.local        # API_BASE_URL=http://localhost:8000
pnpm install
pnpm dev                          # http://localhost:3000
```

Sign in with the admin account. Further users and API keys are managed
under **Settings → Users & API keys**. `NEXT_PUBLIC_USE_MOCKS=1 pnpm dev`
runs the console on bundled demo data without a backend.

## Before you go further

- Every setting is a `VPP_*` environment variable:
  [configuration reference](../configuration.md).
- Protocols, Modbus polling, device control and demand-response
  auto-response are **off** until you enable them.
- Read [security](../security.md) before exposing the API beyond
  localhost.

Next: [Tutorial: simulated inverter](tutorial.md).
