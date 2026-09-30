// PostHog product analytics, cookieless: nothing is stored in the browser, so every page load is
// a new anonymous visitor and no consent banner is needed. Session replay is on (canvas included,
// at a low frame rate) so a claimed race can be watched as well as replayed from its flap log.
// An empty key turns analytics off; events are then printed to the console for local checks.

// Project API key: public by design (it can only send events), safe to commit.
const POSTHOG_KEY = "phc_p6oxVMtDYUnCeaSt6NWm4PQXG5fBAEzhaS3VrUK2u9YN"; // project 289642, EU cloud
const POSTHOG_HOST = "https://eu.i.posthog.com";

let enabled = false;

export function initAnalytics() {
  if (!POSTHOG_KEY || enabled) return;
  // keep local testing out of the real numbers unless asked for
  const local = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname);
  if (local && new URLSearchParams(location.search).get("analytics") !== "1") return;
  // PostHog's standard loader snippet: queues calls until array.js arrives from their asset host.
  // prettier-ignore
  !function(t,e){var o,n,p,r;e.__SV||(window.posthog=e,e._i=[],e.init=function(i,s,a){function g(t,e){var o=e.split(".");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}(p=t.createElement("script")).type="text/javascript",p.crossOrigin="anonymous",p.async=!0,p.src=s.api_host.replace(".i.posthog.com","-assets.i.posthog.com")+"/static/array.js",(r=t.getElementsByTagName("script")[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a="posthog",u.people=u.people||[],u.toString=function(t){var e="posthog";return"posthog"!==a&&(e+="."+a),t||(e+=" (stub)"),e},u.people.toString=function(){return u.toString(1)+".people (stub)"},o="init capture register register_once register_for_session unregister unregister_for_session getFeatureFlag getFeatureFlagPayload isFeatureEnabled reloadFeatureFlags on onFeatureFlags onSessionId identify setPersonProperties group resetGroups reset get_distinct_id get_session_id get_session_replay_url alias set_config startSessionRecording stopSessionRecording sessionRecordingStarted captureException get_property getSessionProperty opt_in_capturing opt_out_capturing has_opted_in_capturing has_opted_out_capturing debug".split(" "),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])},e.__SV=1)}(document,window.posthog||[]);
  window.posthog.init(POSTHOG_KEY, {
    api_host: POSTHOG_HOST,
    persistence: "memory",
    person_profiles: "identified_only",
    respect_dnt: true,
    capture_pageview: true,
    capture_pageleave: true,
    session_recording: {
      maskAllInputs: true,
      captureCanvas: { recordCanvas: true, canvasFps: 2, canvasQuality: "0.4" },
    },
  });
  enabled = true;
}

/**
 * @param {string} event
 * @param {object} [props]
 * @param {{beacon?: boolean}} [opts] beacon: the page is closing, so send with navigator.sendBeacon
 */
export function track(event, props = {}, { beacon = false } = {}) {
  if (!enabled) {
    console.debug("[analytics off]", event, JSON.stringify(props));
    return;
  }
  window.posthog.capture(event, props, beacon ? { transport: "sendBeacon" } : undefined);
}
