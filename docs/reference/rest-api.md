---
hide:
  - navigation
  - toc
---

# REST API (OpenAPI)

The full OpenAPI schema of the API, generated from
`vpp.api.app.create_app()` when this site was built, so it always matches
the code it was built from. Download it as
[openapi.json](openapi.json). A running server serves the same schema at
`/openapi.json` and interactive Swagger UI at `/docs`.

Roles, authentication, pagination, the WebSocket protocol and worked
examples are in the [API guide](../api.md).

<div id="redoc-container" class="vpp-redoc"></div>
<noscript>This page needs JavaScript to render the schema; <a href="../openapi.json">openapi.json</a> is the raw file.</noscript>

<!-- ReDoc is served from this site: scripts/mkdocs_hooks.py takes it from the pinned npm release at build time. -->
<script src="../redoc/redoc.standalone.js"></script>
<script>
  Redoc.init(
    "../openapi.json",
    {
      scrollYOffset: ".md-header",
      hideHostname: true,
      expandResponses: "200,201",
      jsonSampleExpandLevel: 2,
      theme: { typography: { fontFamily: "inherit", headings: { fontFamily: "inherit" } } },
    },
    document.getElementById("redoc-container"),
  );
</script>
