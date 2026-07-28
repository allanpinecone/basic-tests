# Bulk Import into Pinecone BYOC — Troubleshooting Guide

A field guide for diagnosing failed bulk imports into a **BYOC (Bring Your Own Cloud)**
index. Written for AWS/S3, with notes for GCP and Azure where they differ.

If you are staring at a generic error message, start at [Step 0](#step-0-turn-the-generic-error-into-a-real-error)
— do not start by changing IAM policies.

---

## The short version

Most BYOC import failures come down to one of these, roughly in order of how often
they bite:

| # | Root cause | Fast check |
|---|-----------|-----------|
| 1 | The **BYOC cluster's** IAM identity cannot read the bucket (missing `s3:GetObject` / `s3:ListBucket`, or missing `kms:Decrypt` on an SSE-KMS bucket) | [Section 3](#3-s3-permissions--who-actually-needs-them) |
| 2 | **No network path** from the cluster's subnets to S3 (no NAT gateway, no S3 VPC endpoint, or a restrictive endpoint/bucket policy) | [Section 4](#4-network-routing--the-three-paths-that-matter) |
| 3 | The bucket lives in a **different cloud account** than the BYOC deployment (documented limitation for private buckets) | [Section 3](#cross-account-buckets) |
| 4 | Parquet files are **not nested in a namespace subdirectory**, or the `uri` points at a file instead of a prefix | [Section 5](#5-bucket-layout-and-parquet-requirements) |
| 5 | The **namespace already exists** — very common on a *retry* after a partly successful first attempt | [Section 6](#6-namespace-rules-the-retry-trap) |
| 6 | Bucket is in a **different region** than the BYOC deployment, so the S3 gateway endpoint doesn't cover it | [Section 4](#path-b-byoc-cluster--s3-the-actual-data-path) |
| 7 | Wrong **cloud pairing** — you cannot import from S3 into a GCP- or Azure-hosted index | [Section 7](#7-limits-and-unsupported-configurations) |

The single most important thing to internalize: **your script does not read the bucket.
Your BYOC cluster does.** See the next section.

---

## Step 0: Turn the generic error into a real error

Pinecone returns a specific failure reason on the import record itself. A generic
message in your terminal usually means the script printed the exception instead of
the import's `error` field, or the import failed *after* it was accepted.

First, separate the two very different failure classes:

**Class A — `start_import` itself throws.** Nothing was queued. This is a client-side
problem: auth, DNS, TLS, routing to the index host, a malformed `uri`, or a rejected
request. Confirm with `list_imports` — if no new import appears, you are in Class A.
Go to [Section 2](#2-where-to-run-the-import-script-from).

**Class B — the import is accepted, then reports `Failed`.** The cluster tried to do
the work and could not. Read the `error` field. Go to sections 3–6.

### Read the real error

```python
from pinecone import Pinecone

pc = Pinecone(api_key="...")
index = pc.Index(host="https://YOUR-INDEX.svc.YOUR-ENV.byoc.pinecone.io")

desc = index.describe_import(id="YOUR_IMPORT_ID")
print(desc)                      # full object, includes the error field on failure
```

Or with curl, which is useful because it shows the HTTP status and raw body:

```bash
curl -sS -i -X GET "https://$INDEX_HOST/bulk/imports/$IMPORT_ID" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "X-Pinecone-Api-Version: 2025-10"
```

A successful response looks like this, and gains an `error` field when `status` is `Failed`:

```json
{
  "id": "101",
  "uri": "s3://example_bucket/import",
  "status": "InProgress",
  "created_at": "2024-08-19T20:49:00.754Z",
  "percent_complete": 42.2,
  "records_imported": 1000000
}
```

`status` progresses through `Pending` → `InProgress` → `Completed`, or ends in
`Failed` / `Cancelled`.

Also check **Console → your index → Imports tab**, which shows the same failure
reason, and `list_imports` to see whether earlier attempts left partial state behind
(this matters a lot — see [Section 6](#6-namespace-rules-the-retry-trap)).

If you use the toolkit in this repo, the verbose monitor dumps every field returned
by the API each poll, which is the quickest way to see an `error` you didn't know
was there:

```bash
python pinecone_toolkit.py     # Bulk Import → 3. Check import status by ID → Verbose: y
```

### Errors that mean "your data or request", verbatim

If you see any of these, you have a data/layout problem, not a networking problem:

```
User error: The namespace "example-namespace" already exists. Imports are only allowed into nonexistent namespaces.
User error: "test-import/0.parquet": No namespace detected. Each file should be nested under a subdirectory of the URI prefix.
User error: No Parquet files found under "gs://example_bucket/imports". Files must be stored with the specified bucket prefix.
User error: "test-import/0.parquet": It looks like you specified a complete path to a parquet file as the URI prefix to import from.
User error: No vectors added, all rows were skipped for namespace: example-namespace
Missing required column "{0}"
Unsupported column "{0}"
Parquet footer could not be parsed. Are you sure this is valid parquet?
The expected data type for column "{column}" is "{expected}", but got "{given}"
```

If instead the error is vague about storage access, or the import fails almost
immediately with nothing data-specific, treat it as permissions or routing.

---

## 1. How a BYOC import actually works

Understanding the actors resolves most confusion, especially about where to run
things and whose credentials matter.

```
┌──────────────────────────┐
│  Your machine / CI job   │   Runs the script. Calls start_import + describe_import.
│  (the "import script")   │   Needs: HTTPS to the index host. NO S3 access needed.
└────────────┬─────────────┘
             │  (A) HTTPS 443 to index data plane
             ▼
┌──────────────────────────────────────────────┐
│  BYOC data plane — YOUR cloud account, YOUR  │
│  VPC (EKS/GKE/AKS + object storage + DB)     │
│                                              │
│  This is what reads your Parquet files.      │──(B) HTTPS 443──▶ ┌──────────────┐
│  Uses ITS OWN cloud identity, not yours.     │                   │  Source S3   │
└────────────┬─────────────────────────────────┘                   │   bucket     │
             │  (C) outbound 443, agent pull                       └──────────────┘
             ▼
┌──────────────────────────┐
│ Pinecone control plane   │   Global, managed by Pinecone. Index lifecycle, auth,
│ (api.pinecone.io)        │   billing. Never stores or processes your vectors.
└──────────────────────────┘
```

Three consequences that trip people up:

1. **Your laptop's AWS credentials are irrelevant to whether the import succeeds.**
   They only matter for pre-flight validation you run yourself (like the toolkit's
   "Validate storage path" option). It is entirely normal — and a classic red
   herring — for validation to pass from your laptop while the import fails,
   because the cluster has different permissions and a different network path.
2. **`start_import` is a data-plane call** against the index host, so the script must
   be able to reach the index endpoint. Control-plane calls (`list_indexes`,
   `create_index`) always go to `api.pinecone.io` over the public internet, so
   "listing indexes works" does **not** prove the data plane is reachable.
3. **No inbound access to your VPC is required** and Pinecone never gets direct
   access to your cloud account. The cluster pulls work outbound. So there is no
   Pinecone-side IP to allowlist for the import to read your bucket — the reader is
   your own cluster.

---

## 2. Where to run the import script from

### Does the region matter?

For **correctness, no** — the script only issues API calls. Run it from your laptop,
an EC2 instance, a CI runner, or a bastion host. Region affects latency only.

For **the bucket, yes** — the bucket's account and region matter a great deal, because
the cluster is the one reading it. See sections 3 and 4.

### Does the VPC matter?

That depends entirely on how the BYOC environment was deployed:

| BYOC network mode | Where the script can run | Host to target |
|---|---|---|
| `public-access-enabled: true` (default) | Anywhere with internet access | `host` — e.g. `https://my-index-abc123.svc.us-east-1.byoc.pinecone.io` |
| `public-access-enabled: false` (private only) | **Only inside your VPC**, or a network with routed access to it (peering, Transit Gateway, VPN, Direct Connect) | `private_host` — e.g. `https://my-index-abc123.svc.private.us-east-1.byoc.pinecone.io` |

Get both values from `describe_index` (`host` and `private_host`) or the console.
The only difference in the URL is `.svc.` becoming `.svc.private.`.

For private-only mode you must also have completed the private connectivity setup:

- **AWS** — a VPC interface endpoint (PrivateLink) to the service name from the Pulumi
  stack outputs, in the BYOC VPC, with **Enable DNS name** turned on.
- **GCP** — a Private Service Connect endpoint to the service attachment, plus a private
  DNS zone for `<YOUR-BYOC-ENVIRONMENT>.byoc.pinecone.io` with a wildcard `A` record
  pointing at the endpoint IP.
- **Azure** — a private endpoint to the Private Link Service, plus a private DNS zone
  named `<YOUR-BYOC-ENVIRONMENT>.byoc.pinecone.io` linked to the VNet, with a wildcard
  `A` record.

### The private-host gotcha (a genuinely confusing failure)

We have observed BYOC private hostnames resolving in DNS **from outside the VPC** to an
RFC1918 address that is only routable inside the VPC. The result: DNS looks healthy,
`nslookup` returns an answer, and then the connection hangs and times out with a
non-specific error. It looks like Pinecone is down; it is actually a routing problem.

Verify with a real TCP connect, not DNS:

```bash
# DNS resolving is NOT proof of reachability
nslookup my-index-abc123.svc.private.us-east-1.byoc.pinecone.io

# This is the real test
nc -vz -w 5 my-index-abc123.svc.private.us-east-1.byoc.pinecone.io 443
curl -sS -o /dev/null -w '%{http_code}\n' --max-time 10 \
  "https://my-index-abc123.svc.private.us-east-1.byoc.pinecone.io/describe_index_stats" \
  -H "Api-Key: $PINECONE_API_KEY"
```

`byoc_bulk_import.py` in this repo does this probe automatically: it only rewrites a
public host to `.svc.private.` if a TCP connect on 443 actually succeeds, and it
prints which endpoint it chose.

Also confirm the **security group** on the VPC endpoint allows inbound 443 from the
client's security group or CIDR — an otherwise perfect PrivateLink setup fails here.

### Practical recommendation

For private-only BYOC deployments, run imports from a small EC2 instance (or a pod, or
a CI runner with VPC access) in a **private subnet of the BYOC VPC**. That removes DNS,
routing, and security group ambiguity from the equation in one move, and it is also
the environment where any pre-flight bucket validation you run will most closely match
what the cluster itself can see.

---

## 3. S3 permissions — who actually needs them

### The identity that matters is your cluster's, not Pinecone's

In BYOC, the process reading your Parquet files runs on your own Kubernetes cluster in
your own account. It authenticates with the identity your BYOC deployment was
provisioned with — on AWS, IAM roles created by the deployment (IRSA / node role); on
GCP, service accounts with Workload Identity; on Azure, managed identities with
Workload Identity.

> **Important distinction.** Pinecone's **storage integration** flow — where you create
> an IAM role that trusts Pinecone's AWS account (`713131977538`) and pass an
> `integration_id` to `start_import` — is the mechanism for the **standard SaaS**
> service, where Pinecone's infrastructure does the reading. It is documented as
> *optional for public buckets*. For BYOC, the reader is your cluster, and the toolkit
> in this repo deliberately sends **no** `integration_id` for BYOC environments. If you
> have been passing an `integration_id` to a BYOC index in the hope of fixing an access
> error, that is very likely not the lever — verify with Pinecone support which identity
> your specific BYOC environment uses to read a *source* bucket before investing more
> time in cross-account role setup.

### Minimum S3 actions required

Bulk import needs exactly two things: list the prefix, and read the objects.

- `s3:ListBucket` — on the **bucket** ARN, required to enumerate the import directory.
- `s3:GetObject` — on the **object** ARN, required to read each Parquet file.

Scoped to a single import prefix, the correct policy shape is:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "ListImportPrefix",
            "Effect": "Allow",
            "Action": "s3:ListBucket",
            "Resource": "arn:aws:s3:::my-import-bucket",
            "Condition": {
                "StringLike": {
                    "s3:prefix": [
                        "import-data/",
                        "import-data/*"
                    ]
                }
            }
        },
        {
            "Sid": "ReadImportObjects",
            "Effect": "Allow",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::my-import-bucket/import-data/*"
        }
    ]
}
```

Two details cause silent failures here:

- `ListBucket` operates on the **bucket** resource and filters by prefix with a
  `Condition`. Putting `arn:aws:s3:::bucket/prefix/*` as the `Resource` for
  `ListBucket` does not work.
- Include the prefix **both with and without** the trailing wildcard
  (`import-data/` *and* `import-data/*`), or listing the directory itself fails.

Attach this policy to the identity your cluster actually uses (see the discovery
commands in [Section 8](#8-diagnostic-playbook)).

### SSE-KMS encrypted buckets — a top cause of vague errors

If the source objects are encrypted with SSE-KMS, `s3:GetObject` alone is not enough.
The cluster's role needs KMS permissions **and** the KMS key policy must allow that
role:

```json
{
    "Effect": "Allow",
    "Action": [
        "kms:Decrypt",
        "kms:DescribeKey"
    ],
    "Resource": "arn:aws:kms:REGION:ACCOUNT:key/KEY-ID"
}
```

A KMS denial often surfaces as a non-specific read failure rather than a clean
"access denied", which is exactly the sort of generic error that sends people down the
wrong path. If your bucket is encrypted with a customer-managed key, check this early.

### Bucket policy conditions that quietly block the cluster

Even with correct IAM, a bucket policy can deny the request. Audit the source bucket
policy for `Deny` statements and conditions on:

- `aws:SourceVpce` / `aws:SourceVpc` — if the policy requires a specific VPC endpoint
  and the cluster egresses via NAT (a public IP), it is denied.
- `aws:SourceIp` — the cluster's NAT gateway EIPs must be included.
- `aws:PrincipalOrgID`, `aws:PrincipalArn`, `s3:ResourceAccount` — must match the
  cluster's identity and account.
- `aws:SecureTransport` — harmless in practice (the cluster uses TLS), but confirm the
  condition is `"true"` and not inverted.
- **Requester Pays** — if enabled, reads require an explicit requester-pays flag that
  the import does not send. Disable it for the import bucket.

### Cross-account buckets

This is a hard, documented BYOC limitation, and worth checking before anything else:

> Imports from **private cloud storage buckets are not supported unless the bucket is in
> the same cloud account as your BYOC deployment.**

So if the Parquet files live in a different AWS account from the BYOC cluster and the
bucket is private, do not spend time on cross-account bucket policies. Instead:

- Copy or sync the data into a bucket in the **BYOC account** (`aws s3 sync` is fine),
  ideally in the same region; or
- Make the source data public, which is usually unacceptable for production data; or
- Open a support ticket with Pinecone to confirm current support status for your
  environment version.

### GCP and Azure equivalents

- **GCS** — the cluster's service account needs `storage.objects.get` and
  `storage.objects.list` on the bucket (`roles/storage.objectViewer` covers both), plus
  CMEK decrypt rights if the bucket uses customer-managed keys.
- **Azure Blob** — the cluster's managed identity needs `Storage Blob Data Reader` on
  the container, and any storage account firewall must permit the cluster's subnet or
  private endpoint.

---

## 4. Network routing — the three paths that matter

Label the paths and test them independently. Most "generic error" tickets are one
broken path being misattributed to another.

### Path A: script → BYOC index data plane

- Protocol/port: HTTPS 443 to the index host (or `private_host`).
- Covered in [Section 2](#2-where-to-run-the-import-script-from).
- Failure signature: **Class A** — the `start_import` call itself throws, hangs, or
  times out, and no new import appears in `list_imports`.

### Path B: BYOC cluster → S3 (the actual data path)

This is the path that carries your vectors, and the one most often missing. The
cluster's node subnets are typically **private**, so they need one of:

1. **An S3 gateway VPC endpoint** (`com.amazonaws.<region>.s3`) — the usual choice.
   Two constraints that cause confusing partial failures:
   - The endpoint must be **associated with the route tables of the subnets the cluster
     nodes run in**. An endpoint that exists but is attached to the wrong route table
     does nothing.
   - **Gateway endpoints only reach buckets in the same region.** A cross-region source
     bucket will not work through the endpoint and must go via NAT. This is the usual
     explanation for "the import works for one bucket but not another."
   - If the endpoint has a **custom policy**, it must allow `s3:GetObject` and
     `s3:ListBucket` on the source bucket. The default is full access; custom policies
     are frequently too narrow.
2. **A NAT gateway** with a default route from the private subnets to it. The BYOC
   Pulumi deployment provisions NAT gateways, but if the source bucket is reached this
   way, the bucket policy must tolerate the NAT's public IPs (see above).

Verify:

```bash
# Which endpoints exist in the BYOC VPC?
aws ec2 describe-vpc-endpoints --filters "Name=vpc-id,Values=vpc-XXXX" \
  --query 'VpcEndpoints[].{Svc:ServiceName,Type:VpcEndpointType,RouteTables:RouteTableIds,Policy:PolicyDocument}'

# Do the node subnets actually route to NAT or the endpoint?
aws ec2 describe-route-tables --filters "Name=vpc-id,Values=vpc-XXXX" \
  --query 'RouteTables[].{Id:RouteTableId,Assoc:Associations[].SubnetId,Routes:Routes[].{Dst:DestinationCidrBlock,Pfx:DestinationPrefixListId,NAT:NatGatewayId,GW:GatewayId}}'

# Where does the source bucket actually live?
aws s3api get-bucket-location --bucket my-import-bucket
```

Also check any **egress firewall, proxy, or NetworkPolicy** in the cluster. If outbound
traffic is filtered by domain, allowlist `s3.<region>.amazonaws.com` and
`my-import-bucket.s3.<region>.amazonaws.com` (or the GCS/Azure Blob equivalents).

Failure signature: **Class B** — the import is accepted, then fails or sits at 0%
progress with no records imported.

### Path C: BYOC cluster → Pinecone control plane

The in-cluster agent pulls work and reports status outbound over 443. If this is
broken, the environment shows as disconnected and imports may never progress. Confirm
cluster health:

```bash
kubectl get pods -A | grep -E "(pinecone|pc-)"    # all should be Running
kubectl get cluster-operations                     # operations audit trail
```

Any pod stuck in `Pending` or `CrashLoopBackOff` is a cluster-health issue to resolve
before blaming the import.

### What you do *not* need

- No inbound access from Pinecone into your VPC.
- No Pinecone-owned IP allowlisted on the source bucket for a BYOC import.
- No S3 credentials on the machine running the script (for the import itself).

---

## 5. Bucket layout and Parquet requirements

### Directory structure

Every Parquet file must sit in a **namespace subdirectory** under the prefix you pass
as `uri`. The subdirectory name becomes the namespace.

```
my-import-bucket/
└── import-data/                 ◀── this is your uri: s3://my-import-bucket/import-data
    ├── example_namespace1/
    │   ├── 0.parquet
    │   ├── 1.parquet
    │   └── 2.parquet
    └── example_namespace2/
        ├── 3.parquet
        └── 4.parquet
```

- `uri` must name the **directory that contains the namespace folders** — not a
  namespace folder, and never an individual `.parquet` file.
- Files sitting directly under the prefix produce: `No namespace detected. Each file
  should be nested under a subdirectory of the URI prefix.`
- Pointing `uri` at a file produces: `It looks like you specified a complete path to a
  parquet file as the URI prefix to import from.`
- To import into the default namespace, use a subdirectory literally named
  `__default__`, and the index's default namespace must be empty.
- Non-Parquet files under the prefix are ignored, but a namespace folder with no
  Parquet files fails with `No Parquet files found under ...`.

The toolkit validates exactly this layout before you spend 10+ minutes on an import:

```bash
python pinecone_toolkit.py     # Bulk Import → 5. Validate storage path
```

Remember the caveat from [Section 1](#1-how-a-byoc-import-actually-works): validation
runs with **your** credentials from **your** network, so it proves the layout is right
and tells you nothing definitive about the cluster's access.

### Parquet schema

The file must contain exactly the expected columns — **extra columns cause a failure**.

For an index of dense vectors:

| Column | Parquet type | Required |
|---|---|---|
| `id` | `STRING` | Yes |
| `values` | `LIST<FLOAT>` | Yes |
| `metadata` | `STRING` (JSON-encoded, UTF-8) | Optional, `NULL` to omit |

For an index of sparse vectors, replace `values` with
`sparse_values` (`STRUCT<indices: LIST<UINT_32>, values: LIST<FLOAT>>`), required. For a
hybrid index, include `values` (required) and `sparse_values` (optional).

Additional checks worth doing before an import:

- **Dimension must match the index.** A mismatch produces per-record errors that, in
  `CONTINUE` mode, can end as `No vectors added, all rows were skipped`.
- **`metadata` is a JSON-encoded string**, not a nested Parquet struct. The error is
  explicit: `The expected data type for metadata is a JSON encoded string in UTF-8
  format, but got "{given}"`.
- **Vector type must match the index** — pushing dense vectors at a sparse-only index
  fails with `Upserting dense vectors is not supported for indexes that store only
  sparse vectors`.
- **Duplicate vectors** (identical values) are skipped, not imported. With `ABORT` the
  import *fails* on the first duplicate; with `CONTINUE` duplicates are skipped
  silently. If your record count comes in lower than expected and nothing errored,
  suspect duplicates.

Inspect a file locally before uploading:

```bash
python pinecone_toolkit.py     # 7. Inspect Parquet File
```

### Error mode, and why `ABORT` is better for a first run

- `ImportErrorMode.ABORT` — stops at the first bad record and tells you the **file name
  and row number**. Use this while debugging.
- `ImportErrorMode.CONTINUE` — skips bad records and keeps going, but **gives no
  notification about which records failed**. Use this for production bulk loads once
  the data shape is proven.

If a customer reports "the import succeeded but records are missing," it is almost
always `CONTINUE` mode silently skipping invalid or duplicate rows. Compare
`records_imported` from `describe_import` against the expected row count.

---

## 6. Namespace rules (the retry trap)

Two rules that generate a lot of confusing second-attempt failures:

- **You cannot import into an existing namespace.** Imports only target namespaces that
  do not yet exist.
- **Importing into `__default__` requires the default namespace to be empty.**

The trap: attempt #1 partially succeeds and creates namespace `foo`. You fix something
and retry with the same layout. Attempt #2 now fails with:

```
User error: The namespace "foo" already exists. Imports are only allowed into nonexistent namespaces.
```

The failure has nothing to do with your fix, which makes it look like the original
problem was never resolved. Before each retry:

```python
print(index.describe_index_stats())   # which namespaces already exist?
```

Then either delete the leftover namespace or rename the subdirectory in the bucket. The
toolkit exposes namespace deletion at **Bulk Import → 6. Delete a namespace**.

Also remember every import takes **at least 10 minutes**, and imported vectors can take
another ~10 minutes to become queryable after the import reports `Completed`. An empty
`describe_index_stats` immediately after completion is not a failure.

---

## 7. Limits and unsupported configurations

Exceeding a limit yields an error naming the limit, so check these against the dataset:

| Metric | Limit |
|---|---|
| Max namespaces per import | 10,000 |
| Max size per namespace | 500 GB |
| Max total input data size (on-demand indexes) | 1 TB |
| Max files per import | 100,000 |
| Max size per file | 10 GB |

The 1 TB total does not apply to indexes with **dedicated read nodes**, which support
larger imports. This is relevant to BYOC, which supports **DRN indexes only** (not
on-demand indexes).

Hard constraints to rule out:

- **You cannot import from an S3 bucket into an index hosted on GCP or Azure.** The
  supported pairings are:

  | | → AWS index | → GCP index | → Azure index |
  |---|:---:|:---:|:---:|
  | From **AWS S3** | ✅ | ❌ | ❌ |
  | From **Google Cloud Storage** | ✅ | ✅ | ✅ |
  | From **Azure Blob Storage** | ✅ | ✅ | ✅ |

  For a BYOC-on-AWS environment, use `s3://`. The toolkit warns on a scheme/environment
  mismatch before starting.
- **S3 Express One Zone is not supported** as a source.
- **Imports from private buckets outside the BYOC cloud account are not supported.**
- **Bulk import requires an index without a schema definition.** It is not supported for
  indexes with schemas, including full-text-search indexes with document schemas and
  semantic-text-only integrated-embedding indexes.
- **Integrated-embedding indexes require vectors, not text**, in the Parquet files.
- Bulk import works with **serverless indexes** only and is in **public preview**,
  available on Standard and Enterprise plans.

---

## 8. Diagnostic playbook

Run these in order and record the output. This is also exactly what a support ticket
needs.

### A. Identify the environment

```python
from pinecone import Pinecone
pc = Pinecone(api_key="...")
print(pc.describe_index("YOUR_INDEX"))   # note: host, private_host, spec.byoc.environment, status
```

### B. Prove Path A (script → data plane)

```bash
nc -vz -w 5 "$INDEX_HOSTNAME" 443
curl -sS -o /dev/null -w 'HTTP %{http_code} in %{time_total}s\n' --max-time 15 \
  "https://$INDEX_HOSTNAME/describe_index_stats" -H "Api-Key: $PINECONE_API_KEY"
```

The toolkit runs a staged DNS → TCP → HTTPS check at
**Index Management → 3. Test connectivity**, which is useful because it tells you *which*
stage fails.

### C. Confirm the source layout and region

```bash
aws s3 ls s3://my-import-bucket/import-data/ --recursive --human-readable --summarize
aws s3api get-bucket-location --bucket my-import-bucket
aws s3api get-bucket-encryption --bucket my-import-bucket        # SSE-KMS?
aws s3api get-bucket-policy --bucket my-import-bucket | jq -r '.Policy | fromjson'
aws s3api get-bucket-request-payment --bucket my-import-bucket   # Requester Pays?
```

### D. Find the cluster's identity, then test *its* access

Discover the IAM role the Pinecone workloads use (IRSA annotation):

```bash
kubectl get sa -A -o json \
  | jq -r '.items[] | select(.metadata.annotations["eks.amazonaws.com/role-arn"]) 
           | "\(.metadata.namespace)/\(.metadata.name)\t\(.metadata.annotations["eks.amazonaws.com/role-arn"])"'
```

Then simulate that principal's access to the exact objects:

```bash
ROLE_ARN=arn:aws:iam::ACCOUNT:role/THE-ROLE-FROM-ABOVE

aws iam simulate-principal-policy \
  --policy-source-arn "$ROLE_ARN" \
  --action-names s3:ListBucket \
  --resource-arns arn:aws:s3:::my-import-bucket \
  --query 'EvaluationResults[].{Action:EvalActionName,Decision:EvalDecision}'

aws iam simulate-principal-policy \
  --policy-source-arn "$ROLE_ARN" \
  --action-names s3:GetObject \
  --resource-arns "arn:aws:s3:::my-import-bucket/import-data/example_namespace1/0.parquet" \
  --query 'EvaluationResults[].{Action:EvalActionName,Decision:EvalDecision}'
```

`implicitDeny` or `explicitDeny` here is your answer. Note that `simulate-principal-policy`
does not evaluate KMS key policies, so check those separately.

### E. Prove Path B from inside the cluster's network

Launch a short-lived debug pod in your own namespace (not a Pinecone one) on the same
subnets and confirm it can reach S3:

```bash
kubectl run s3-netcheck --rm -it --restart=Never --image=amazonlinux:2023 -- bash -lc '
  dnf install -y awscli-2 >/dev/null 2>&1 || dnf install -y awscli >/dev/null 2>&1
  echo "--- DNS ---";  getent hosts my-import-bucket.s3.us-east-1.amazonaws.com
  echo "--- TLS ---";  curl -sS -o /dev/null -w "HTTP %{http_code}\n" --max-time 10 https://my-import-bucket.s3.us-east-1.amazonaws.com
  echo "--- LIST ---"; aws s3 ls s3://my-import-bucket/import-data/ --region us-east-1
'
```

A hang or timeout on TLS means Path B is broken (no NAT / no usable endpoint). An HTTP
403 means the path works and the problem is permissions — a very useful distinction.

### F. Re-run the import with maximum signal

Use `ABORT` mode and a **fresh namespace name** so you get a precise, first-failure error:

```bash
python byoc_bulk_import.py \
  --host "https://my-index-abc123.svc.us-east-1.byoc.pinecone.io" \
  --uri  "s3://my-import-bucket/import-data" \
  --error-mode abort \
  --poll-interval 10
```

---

## 9. What to send Pinecone support

If everything above checks out, open a ticket with:

1. BYOC **environment name** and cloud/region (e.g. `aws-us-east-1-26bf.byoc`).
2. Index name, and whether you are using `host` or `private_host`.
3. The **import ID** and the **full `describe_import` response**, including the `error`
   field, verbatim.
4. The `uri` you passed, and an `aws s3 ls --recursive` listing of that prefix.
5. Bucket **account ID and region** vs. the BYOC deployment's account ID and region.
6. Whether the bucket is public or private, and whether it uses SSE-KMS.
7. The IAM role ARN you believe the cluster uses, plus the
   `simulate-principal-policy` results.
8. Output of `kubectl get pods -A | grep -E "(pinecone|pc-)"` and
   `kubectl get cluster-operations`.
9. Confirmation of whether Path B was reachable from a debug pod (Step E above).

Explicitly ask support to confirm **which cloud identity your BYOC data plane uses to
read a source import bucket**, and whether an `integration_id` is expected for your
environment version. That is the one piece that is not fully pinned down in the public
docs, and it determines where the permissions must be attached.

---

## Reference

- [Import data](https://docs.pinecone.io/guides/index-data/import-data) — format, limits, error catalog
- [Understanding imports](https://docs.pinecone.io/guides/index-data/understanding-imports) — directory structure
- [Bring your own cloud (BYOC)](https://docs.pinecone.io/guides/production/bring-your-own-cloud) — architecture, network modes, limitations
- [Integrate with Amazon S3](https://docs.pinecone.io/guides/operations/integrations/integrate-with-amazon-s3) — the SaaS storage integration flow
- [Describe an import](https://docs.pinecone.io/reference/api/latest/data-plane/describe_import) — status and error fields
- `byoc_bulk_import.py` — standalone import script with private-endpoint detection
- `pinecone_toolkit.py` — interactive connectivity tests, storage validation, Parquet inspection
