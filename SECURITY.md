# Security Policy

## Artefact Verification

Commits and tags in this repository SHALL be signed with the maintainer key published in `KEYS.asc`.

| Field | Value |
|---|---|
| Algorithm | Ed25519 (OpenPGP) |
| Key ID | `2CC9231D80106E16` |
| Fingerprint | `E4E4 3B59 CAD0 3044 D381 7449 2CC9 231D 8010 6E16` |
| Identity | Dhruv Setty, `91457220+whatheheckisthis@users.noreply.github.com` |
| Expiry | 2028-11-13 |

To verify a commit or tag:

```bash
gpg --import KEYS.asc
git verify-commit HEAD
git verify-tag <tag>
```

A commit that does not verify against this key SHOULD NOT be treated as originating from the maintainer.

## Reporting a Vulnerability

Vulnerabilities SHALL be reported privately through GitHub Security Advisories on this repository. Public issues MUST NOT be used for undisclosed vulnerabilities.
