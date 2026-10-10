# Releasing

How a maintainer cuts a release of `virtual-power-plant` (PyPI) and the
`ghcr.io/vinerya/virtual-power-plant` API image. Pushing a `v*` tag runs
[`.github/workflows/release.yml`](../.github/workflows/release.yml), which
publishes everything; the steps below get the repository ready for that tag
and check the result.

Versions follow [Semantic Versioning](https://semver.org/): a release with
anything under **Breaking changes** in the changelog bumps the major
version. Pre-releases use PEP 440 suffixes (`3.1.0rc1`, tag `v3.1.0rc1`);
they are published to PyPI and GHCR but never tagged `latest`.

## One-time setup

Done once per repository; skip if already in place.

1. **PyPI trusted publisher.** On
   <https://pypi.org/manage/project/virtual-power-plant/settings/publishing/>
   add a GitHub publisher: owner `vinerya`, repository
   `virtual-power-plant`, workflow `release.yml`, environment `pypi`. (If
   the project does not exist on PyPI yet, add it as a *pending publisher*
   from <https://pypi.org/manage/account/publishing/>.) No API token is
   stored in GitHub; the workflow authenticates with a short-lived OIDC
   token. Delete any old `PYPI_TOKEN` secret and revoke the token on PyPI.
2. **GitHub environment `pypi`.** Settings → Environments → New
   environment `pypi`. Optionally add required reviewers (the publish job
   then waits for approval) and a deployment rule allowing only `v*` tags.
3. **Workflow permissions.** Settings → Actions → General → Workflow
   permissions must let workflows request write scopes (the jobs ask for
   `packages: write` and `contents: write` themselves; the default token is
   read-only).
4. **GHCR visibility.** After the first image push, open the package under
   the repository's *Packages* and make it public if anonymous
   `docker pull` should work; link it to the repository if GitHub did not
   do so automatically.

## Cutting a release

Example: 3.0.0. Work on a branch and merge through a pull request as usual,
so CI runs on the release commit.

1. **Bump the version** in `pyproject.toml` (`[project] version`), the
   single source of truth: `vpp.__version__`, `vpp --version`, `/health`,
   `/version` and the OpenAPI document all read it. Also update `version`
   and `date-released` in `CITATION.cff`.
2. **Finish the changelog.** In `CHANGELOG.md`, rename `## [Unreleased]` to
   `## [3.0.0] - YYYY-MM-DD` and add a new empty `## [Unreleased]` above it.
   Deduplicate and order the entries (Breaking changes, Added, Changed,
   Deprecated, Removed, Fixed, Security), and for a major release write an
   **Upgrading from 2.x** subsection: migrations to run, settings to change,
   breaking API changes, behaviour changes. The release workflow publishes
   this section verbatim as the GitHub Release notes and refuses to run if
   it is missing or empty.
3. **Check locally** (from the repository root):

   ```bash
   ruff check src tests examples && ruff format --check src tests examples
   python -m mypy src
   python -m pytest -q
   rm -rf dist && python -m build && twine check --strict dist/*
   # Install the wheel into a fresh venv outside the checkout:
   python -m venv /tmp/vpp-rel && /tmp/vpp-rel/bin/pip install "$(ls dist/*.whl)[api,db,cli]"
   (cd /tmp && /tmp/vpp-rel/bin/vpp --version \
     && VPP_DATABASE_URL=sqlite+aiosqlite:////tmp/vpp-rel.db /tmp/vpp-rel/bin/vpp migrate)
   ```

   `vpp --version` must print the new version, and `vpp migrate` must reach
   the newest revision (the migrations ship inside the wheel). Do not
   commit `dist/`.
4. **Merge** the release pull request into `main` and wait for CI to pass.
5. **Tag and push** the merge commit:

   ```bash
   git switch main && git pull --ff-only
   git tag -a v3.0.0 -m "v3.0.0"
   git push origin v3.0.0
   ```

   The tag must be exactly `v` + the `pyproject.toml` version.

## What the workflow does

| Job | What it does |
|---|---|
| `ci` | Runs the whole of `ci.yml` (lint, mypy, tests on 3.10-3.12, migrations against PostgreSQL, bandit, Docker build and health check) against the tag. |
| `verify` | Fails unless the tag equals `v<pyproject version>`; extracts the matching `CHANGELOG.md` section (fails if missing or empty); marks PEP 440 pre-releases. |
| `build` | `python -m build`, `twine check --strict`, installs the wheel into a clean venv outside the checkout, checks `vpp --version` and `vpp.__version__`, runs `vpp migrate` against a temporary SQLite file. |
| `publish-pypi` | Needs `ci`, `verify` and `build`. Uploads the built sdist and wheel with `pypa/gh-action-pypi-publish` using trusted publishing (environment `pypi`, `id-token: write`), with PEP 740 attestations. |
| `publish-docker` | Needs `ci`, `verify` and `build`. Builds the root `Dockerfile` and pushes `ghcr.io/vinerya/virtual-power-plant:<version>` and, unless a pre-release, `:latest`, with OCI labels (source, revision, version, licence, ...). |
| `github-release` | After both publish jobs: creates the GitHub Release for the tag with the changelog section as its body and the sdist and wheel attached (marked pre-release or latest). |

Nothing is published unless `ci`, `verify` and `build` all pass. If a
publish job fails (for example a PyPI outage), fix the cause and use
**Re-run failed jobs** on the same workflow run; PyPI refuses to upload a
file that already exists, so a re-run never publishes two different files
under one version.

## Verifying a release

```bash
python -m venv /tmp/check && /tmp/check/bin/pip install "virtual-power-plant[api,db,cli]==3.0.0"
/tmp/check/bin/vpp --version                 # vpp, version 3.0.0
docker pull ghcr.io/vinerya/virtual-power-plant:3.0.0
docker run --rm -d -p 8000:8000 -e VPP_SECRET_KEY="$(openssl rand -hex 32)" \
  --name vpp-check ghcr.io/vinerya/virtual-power-plant:3.0.0
curl -s localhost:8000/health                # "version": "3.0.0"
docker stop vpp-check
```

Also check that <https://pypi.org/project/virtual-power-plant/> shows the
version and its README, that the GitHub Release has the changelog notes
and both distribution files, and that `:latest` points at the new image
(`docker buildx imagetools inspect ghcr.io/vinerya/virtual-power-plant:latest`).

## Yanking and rolling back

A published version can never be replaced: PyPI does not accept a new
file under an existing version, even after deletion. Fix forward with a
patch release (`3.0.1`) and, if the broken version is harmful, also:

- **Yank it on PyPI** (project → *Manage* → *Releases* → `3.0.0` →
  *Options* → *Yank*, with a reason). `pip install virtual-power-plant`
  then skips it, while pins to `==3.0.0` keep working. Prefer yanking over
  deleting; deleting breaks pinned installs and the version still cannot be
  reused.
- **Move `latest` back on GHCR** to the previous good image, e.g.
  `docker buildx imagetools create -t ghcr.io/vinerya/virtual-power-plant:latest ghcr.io/vinerya/virtual-power-plant:<previous-version>`
  (requires `docker login ghcr.io` with a token that has
  `write:packages`). Leave the versioned tag in place, or delete that
  version from the package page if it must not be pulled at all.
- **Mark the GitHub Release** as a pre-release or edit its notes to point
  at the fix; delete it only if it was created by mistake.
- **A tag pushed by mistake** (before anything was published, e.g. the
  `verify` job failed): delete it locally and remotely, fix, and tag
  again.

  ```bash
  git tag -d v3.0.0 && git push origin :refs/tags/v3.0.0
  ```

  Once PyPI has the version, do not reuse the tag; release the next
  patch version instead.
- **Database migrations** are not rolled back by downgrading the package:
  restore the backup taken before `vpp migrate` (data-bearing alembic
  downgrades are not tested, see [deployment.md](deployment.md)).
