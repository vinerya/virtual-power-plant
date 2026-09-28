"""Sites, customer-portal and telemetry-history services.

Ownership model
---------------
* A **site** (``sites``) is a physical premise with a location; it groups
  resources via ``resources.site_id``.
* A site may be owned by one ``customer``-role user (``sites.owner_id``).
  A customer's *devices* are exactly the resources of the sites they own;
  there is no per-resource owner column, so reassigning a site moves all
  of its devices with it.
* Operator-side roles (admin/operator/viewer/researcher) see everything;
  customers only ever see their own sites/resources, and lookups of anything
  else return 404 (never 403) so object existence is not leaked.
"""
