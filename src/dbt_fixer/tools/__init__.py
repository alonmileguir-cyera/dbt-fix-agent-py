"""Model-facing tools: rooted, read-only repository access.

Nothing in this package (or any sub-module of it) ever exposes a write,
create, or delete capability to a model. See `dbt_fixer.tools.repo_tools`
for the sole toolkit (`RepoTools`) wired to the agent.
"""
