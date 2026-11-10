# This fork

This fork is upstream AIHawk ([feder-cr/Jobs_Applier_AI_Agent](https://github.com/feder-cr/Jobs_Applier_AI_Agent)) plus the features below. Each feature lives in `src/aihawk/fork/` and leaves upstream's files alone, so upstream's updates merge without conflicts.

| Feature | Code | Tests |
|---|---|---|
| OpenRouter as the model gateway; Jev called on TypeSafe directly | `src/aihawk/fork/providers.py` | `tests/test_fork_providers.py` |

## OpenRouter and TypeSafe

With nothing set, AIHawk runs exactly as upstream: OrcaRouter, its key, its client.

### Switching to OpenRouter

Pick one way:

- **Environment (recommended):** set `AIHAWK_PROVIDER=openrouter` and `OPENROUTER_API_KEY`. Nothing to do inside the app, and it holds for every AIHawk folder on the machine.

  Windows (PowerShell; open a new terminal afterwards):

  ```powershell
  setx AIHAWK_PROVIDER openrouter
  setx OPENROUTER_API_KEY "sk-or-..."
  ```

  macOS or Linux (add to `~/.zshrc` or `~/.bashrc`):

  ```bash
  export AIHAWK_PROVIDER=openrouter
  export OPENROUTER_API_KEY="sk-or-..."
  ```

- **In the app:** press `e`, choose **Gateway**, type `openrouter`, then choose **OpenRouter key** and paste the key. The key goes into `openrouter-key` in the AIHAWK folder (`~/.aihawk`), and `provider: openrouter` goes into `aihawk.yaml`. If you skip setting `OPENROUTER_API_KEY`, the first start asks for "Your OpenRouter key." and saves it the same way.

`AIHAWK_PROVIDER` wins over the setting; the Settings screen says so if you try to change it there. Each gateway keeps its own key (`key` stays OrcaRouter's), so switching back and forth never sends one gateway's key to the other.

**What stays the same on OpenRouter:**

- **Requests:** the same chat calls, JSON-schema answers, form tools and Jev requests.
- **Client:** upstream's own client, with its retries and timeout, copied onto OpenRouter's address.
- **Cost:** the same cost counting. OpenRouter's `/models` and `/generation` give prices and charges in the shape AIHawk reads.

**What OpenRouter adds:**

- **Routing:** each chat call asks to be routed only to providers that take all its parameters (`provider.require_parameters`), so no provider drops the tools or the schema.
- **Attribution:** calls carry OpenRouter's `X-Title: AIHawk` header.

**Check the model names.** Upstream's defaults (`deepseek/deepseek-v4.1-flash`, `google/gemini-3.5-flash`, `typesafe/jev-1.14`) must exist on OpenRouter under those exact ids. Look them up at <https://openrouter.ai/models> and change **Forms model**, **Search plan model** or **Scoring model** in Settings if one differs. A wrong id fails with "OpenRouter rejected the request".

### Jev on TypeSafe directly

Jev, the scoring model, answers at `/systemone`. OpenRouter serves that endpoint too, so Jev works through OpenRouter without anything else. To call TypeSafe itself instead:

```powershell
setx TYPESAFE_API_KEY "your-typesafe-key"
```

Or put the key alone in a file named `typesafe-key` in the AIHawk folder.

| `decision_provider` in `aihawk.yaml` (Settings: **Scoring goes to**) | Where Jev's calls go |
|---|---|
| `auto` (default) | TypeSafe directly when there is a TypeSafe key, else the gateway |
| `gateway` | Always the gateway, even with a TypeSafe key |
| `typesafe` | Always TypeSafe; fails, and says so, without a key |

**Address and model name:**

- **Model name:** direct calls drop the `typesafe/` prefix (`typesafe/jev-1.14` is sent as `jev-1.14`). Give **Scoring model** a name without the prefix, such as `jev-latest`, to send it as written.
- **Address:** TypeSafe's address is `https://api.typesafe.ai/v1` unless `typesafe_base_url` in `aihawk.yaml` says otherwise. Check it against TypeSafe's own docs before you rely on it.

If TypeSafe refuses its key, AIHawk forgets a typed one, says so, and scores through the gateway again; the gateway's key is not touched.

**Keys and the browser:** the browser that fills forms is given none of `ORCAROUTER_API_KEY`, `OPENROUTER_API_KEY` or `TYPESAFE_API_KEY`.

## Keeping up with upstream

### Branches

| Branch | Holds | Rule |
|---|---|---|
| `main` | upstream's `main`, nothing else | Only ever fast-forwarded to `upstream/main`. Never commit here. |
| `custom` | `main` + this fork's features | What you install and run. Upstream is merged into it. |
| `feature/<name>` | one new feature, while you build it | Branched from `custom`, merged back into it. |

Remotes: `origin` is your fork (`pritkc/Jobs_Applier_AI_Agent`), `upstream` is `feder-cr/Jobs_Applier_AI_Agent`.

### Bringing upstream in (on the machine where you develop)

```bash
git fetch upstream
git switch main
git merge --ff-only upstream/main
git push origin main
git switch custom
git merge main
python -m pytest -m "not browser"
git push origin custom
```

Use merge, not rebase, for `custom`: it is pushed and pulled on other machines, and a rebase would make every one of them force-reset. Turn on rerere once per machine, so that if you ever resolve a conflict, git remembers the resolution:

```bash
git config --global rerere.enabled true
```

### Getting the update on another machine

```bash
git switch custom
git pull
```

Then reinstall only if `pyproject.toml` changed (`uv pip install -e ".[dev]"`).

### Why merges stay clean

- **One hook:** upstream's code has one line of ours, `import aihawk.fork` in `src/aihawk/cli.py`. Every other fork file is new: `src/aihawk/fork/`, `tests/test_fork_*.py`, this page.
- **Wrapping:** at startup each feature wraps the few upstream functions it needs. A wrapper calls upstream's original and changes only what it must, so upstream's fixes to those functions still reach you.
- **Drift guard:** if upstream renames, removes or reshapes something a feature wraps, AIHawk stops at startup with `UpstreamChanged: aihawk.llm.chat ... update src/aihawk/fork`. `tests/test_fork_contract.py` fails the same way, so you learn about it at merge time, not halfway through an application.
- **Tests stay separate:** upstream's own tests run on upstream's code alone. The fork's tests switch the features on for their module only.

### When the drift guard fires

Open the file it names in `src/aihawk/fork/` and look at what upstream changed (`git log -p main -- <upstream file>`). Adapt the wrapper to the new shape, update the parameter names in the `target(...)` call in `activate()`, and run the tests.

If the hook line itself ever conflicts (upstream edited `cli.main` around it), keep upstream's version and put `import aihawk.fork  # noqa: F401` back as the first line of `main`.

## Adding a feature of your own

1. Start a branch from `custom`:

   ```bash
   git switch custom
   git switch -c feature/<name>
   ```

2. Write `src/aihawk/fork/<name>.py` with an `activate()` that:
   - finds each upstream piece it changes with `target(owner, "name", (parameter names...))`;
   - wraps functions with `rebind(original, wrapper)`, which also catches modules that imported the name;
   - wraps methods and class attributes with `replace(Class, "name", wrapper)`;
   - has each wrapper call the original.
3. Add the module to `FEATURES` in `src/aihawk/fork/__init__.py`.
4. Test it in `tests/test_fork_<name>.py`, with the module-scoped fixture that `tests/test_fork_providers.py` uses (`activate()` before, `deactivate()` after).
5. Add a row to the table at the top of this page.
6. Merge it into `custom`:

   ```bash
   git switch custom
   git merge feature/<name>
   git push origin custom
   ```

Keep upstream's files as they are. When a feature truly cannot be done by wrapping (a change in the middle of a long upstream function), keep the edit to as few lines as possible and note it in the table above, so you know where a conflict can come from.
