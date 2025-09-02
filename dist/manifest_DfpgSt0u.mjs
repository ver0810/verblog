import '@astrojs/internal-helpers/path';
import 'kleur/colors';
import { N as NOOP_MIDDLEWARE_HEADER, k as decodeKey } from './chunks/astro/server_Dv6w92OM.mjs';
import 'clsx';
import 'cookie';
import 'es-module-lexer';
import 'html-escaper';

const NOOP_MIDDLEWARE_FN = async (_ctx, next) => {
  const response = await next();
  response.headers.set(NOOP_MIDDLEWARE_HEADER, "true");
  return response;
};

const codeToStatusMap = {
  // Implemented from tRPC error code table
  // https://trpc.io/docs/server/error-handling#error-codes
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  TIMEOUT: 405,
  CONFLICT: 409,
  PRECONDITION_FAILED: 412,
  PAYLOAD_TOO_LARGE: 413,
  UNSUPPORTED_MEDIA_TYPE: 415,
  UNPROCESSABLE_CONTENT: 422,
  TOO_MANY_REQUESTS: 429,
  CLIENT_CLOSED_REQUEST: 499,
  INTERNAL_SERVER_ERROR: 500
};
Object.entries(codeToStatusMap).reduce(
  // reverse the key-value pairs
  (acc, [key, value]) => ({ ...acc, [value]: key }),
  {}
);

function sanitizeParams(params) {
  return Object.fromEntries(
    Object.entries(params).map(([key, value]) => {
      if (typeof value === "string") {
        return [key, value.normalize().replace(/#/g, "%23").replace(/\?/g, "%3F")];
      }
      return [key, value];
    })
  );
}
function getParameter(part, params) {
  if (part.spread) {
    return params[part.content.slice(3)] || "";
  }
  if (part.dynamic) {
    if (!params[part.content]) {
      throw new TypeError(`Missing parameter: ${part.content}`);
    }
    return params[part.content];
  }
  return part.content.normalize().replace(/\?/g, "%3F").replace(/#/g, "%23").replace(/%5B/g, "[").replace(/%5D/g, "]");
}
function getSegment(segment, params) {
  const segmentPath = segment.map((part) => getParameter(part, params)).join("");
  return segmentPath ? "/" + segmentPath : "";
}
function getRouteGenerator(segments, addTrailingSlash) {
  return (params) => {
    const sanitizedParams = sanitizeParams(params);
    let trailing = "";
    if (addTrailingSlash === "always" && segments.length) {
      trailing = "/";
    }
    const path = segments.map((segment) => getSegment(segment, sanitizedParams)).join("") + trailing;
    return path || "/";
  };
}

function deserializeRouteData(rawRouteData) {
  return {
    route: rawRouteData.route,
    type: rawRouteData.type,
    pattern: new RegExp(rawRouteData.pattern),
    params: rawRouteData.params,
    component: rawRouteData.component,
    generate: getRouteGenerator(rawRouteData.segments, rawRouteData._meta.trailingSlash),
    pathname: rawRouteData.pathname || void 0,
    segments: rawRouteData.segments,
    prerender: rawRouteData.prerender,
    redirect: rawRouteData.redirect,
    redirectRoute: rawRouteData.redirectRoute ? deserializeRouteData(rawRouteData.redirectRoute) : void 0,
    fallbackRoutes: rawRouteData.fallbackRoutes.map((fallback) => {
      return deserializeRouteData(fallback);
    }),
    isIndex: rawRouteData.isIndex,
    origin: rawRouteData.origin
  };
}

function deserializeManifest(serializedManifest) {
  const routes = [];
  for (const serializedRoute of serializedManifest.routes) {
    routes.push({
      ...serializedRoute,
      routeData: deserializeRouteData(serializedRoute.routeData)
    });
    const route = serializedRoute;
    route.routeData = deserializeRouteData(serializedRoute.routeData);
  }
  const assets = new Set(serializedManifest.assets);
  const componentMetadata = new Map(serializedManifest.componentMetadata);
  const inlinedScripts = new Map(serializedManifest.inlinedScripts);
  const clientDirectives = new Map(serializedManifest.clientDirectives);
  const serverIslandNameMap = new Map(serializedManifest.serverIslandNameMap);
  const key = decodeKey(serializedManifest.key);
  return {
    // in case user middleware exists, this no-op middleware will be reassigned (see plugin-ssr.ts)
    middleware() {
      return { onRequest: NOOP_MIDDLEWARE_FN };
    },
    ...serializedManifest,
    assets,
    componentMetadata,
    inlinedScripts,
    clientDirectives,
    routes,
    serverIslandNameMap,
    key
  };
}

const manifest = deserializeManifest({"hrefRoot":"file:///home/ancheng/WebApp/myblog/","cacheDir":"file:///home/ancheng/WebApp/myblog/node_modules/.astro/","outDir":"file:///home/ancheng/WebApp/myblog/dist/","srcDir":"file:///home/ancheng/WebApp/myblog/src/","publicDir":"file:///home/ancheng/WebApp/myblog/public/","buildClientDir":"file:///home/ancheng/WebApp/myblog/dist/client/","buildServerDir":"file:///home/ancheng/WebApp/myblog/dist/server/","adapterName":"","routes":[{"file":"file:///home/ancheng/WebApp/myblog/dist/about/index.html","links":[],"scripts":[],"styles":[],"routeData":{"route":"/about","isIndex":false,"type":"page","pattern":"^\\/about\\/?$","segments":[[{"content":"about","dynamic":false,"spread":false}]],"params":[],"component":"src/pages/about.astro","pathname":"/about","prerender":true,"fallbackRoutes":[],"distURL":[],"origin":"project","_meta":{"trailingSlash":"ignore"}}},{"file":"file:///home/ancheng/WebApp/myblog/dist/blog/index.html","links":[],"scripts":[],"styles":[],"routeData":{"route":"/blog","isIndex":false,"type":"page","pattern":"^\\/blog\\/?$","segments":[[{"content":"blog","dynamic":false,"spread":false}]],"params":[],"component":"src/pages/blog.astro","pathname":"/blog","prerender":true,"fallbackRoutes":[],"distURL":[],"origin":"project","_meta":{"trailingSlash":"ignore"}}},{"file":"file:///home/ancheng/WebApp/myblog/dist/contact/index.html","links":[],"scripts":[],"styles":[],"routeData":{"route":"/contact","isIndex":false,"type":"page","pattern":"^\\/contact\\/?$","segments":[[{"content":"contact","dynamic":false,"spread":false}]],"params":[],"component":"src/pages/contact.astro","pathname":"/contact","prerender":true,"fallbackRoutes":[],"distURL":[],"origin":"project","_meta":{"trailingSlash":"ignore"}}},{"file":"file:///home/ancheng/WebApp/myblog/dist/notes/index.html","links":[],"scripts":[],"styles":[],"routeData":{"route":"/notes","isIndex":false,"type":"page","pattern":"^\\/notes\\/?$","segments":[[{"content":"notes","dynamic":false,"spread":false}]],"params":[],"component":"src/pages/notes.astro","pathname":"/notes","prerender":true,"fallbackRoutes":[],"distURL":[],"origin":"project","_meta":{"trailingSlash":"ignore"}}},{"file":"file:///home/ancheng/WebApp/myblog/dist/index.html","links":[],"scripts":[],"styles":[],"routeData":{"route":"/","isIndex":true,"type":"page","pattern":"^\\/$","segments":[],"params":[],"component":"src/pages/index.astro","pathname":"/","prerender":true,"fallbackRoutes":[],"distURL":[],"origin":"project","_meta":{"trailingSlash":"ignore"}}}],"site":"https://ver0810.github.io","base":"/verblog/","trailingSlash":"ignore","compressHTML":true,"componentMetadata":[["\u0000astro:content",{"propagation":"in-tree","containsHead":false}],["/home/ancheng/WebApp/myblog/src/components/blog/Masthead.astro",{"propagation":"in-tree","containsHead":false}],["/home/ancheng/WebApp/myblog/src/layouts/BlogPost.astro",{"propagation":"in-tree","containsHead":false}],["/home/ancheng/WebApp/myblog/src/pages/posts/[slug].astro",{"propagation":"in-tree","containsHead":true}],["\u0000@astro-page:src/pages/posts/[slug]@_@astro",{"propagation":"in-tree","containsHead":false}],["/home/ancheng/WebApp/myblog/src/components/note/Note.astro",{"propagation":"in-tree","containsHead":false}],["/home/ancheng/WebApp/myblog/src/pages/notes/[...page].astro",{"propagation":"in-tree","containsHead":true}],["\u0000@astro-page:src/pages/notes/[...page]@_@astro",{"propagation":"in-tree","containsHead":false}],["/home/ancheng/WebApp/myblog/src/pages/notes/[...slug].astro",{"propagation":"in-tree","containsHead":true}],["\u0000@astro-page:src/pages/notes/[...slug]@_@astro",{"propagation":"in-tree","containsHead":false}],["/home/ancheng/WebApp/myblog/src/pages/blog.astro",{"propagation":"in-tree","containsHead":true}],["\u0000@astro-page:src/pages/blog@_@astro",{"propagation":"in-tree","containsHead":false}],["/home/ancheng/WebApp/myblog/src/pages/index.astro",{"propagation":"in-tree","containsHead":true}],["\u0000@astro-page:src/pages/index@_@astro",{"propagation":"in-tree","containsHead":false}],["/home/ancheng/WebApp/myblog/src/pages/notes.astro",{"propagation":"in-tree","containsHead":true}],["\u0000@astro-page:src/pages/notes@_@astro",{"propagation":"in-tree","containsHead":false}],["/home/ancheng/WebApp/myblog/src/pages/about.astro",{"propagation":"none","containsHead":true}]],"renderers":[],"clientDirectives":[["idle","(()=>{var l=(n,t)=>{let i=async()=>{await(await n())()},e=typeof t.value==\"object\"?t.value:void 0,s={timeout:e==null?void 0:e.timeout};\"requestIdleCallback\"in window?window.requestIdleCallback(i,s):setTimeout(i,s.timeout||200)};(self.Astro||(self.Astro={})).idle=l;window.dispatchEvent(new Event(\"astro:idle\"));})();"],["load","(()=>{var e=async t=>{await(await t())()};(self.Astro||(self.Astro={})).load=e;window.dispatchEvent(new Event(\"astro:load\"));})();"],["media","(()=>{var n=(a,t)=>{let i=async()=>{await(await a())()};if(t.value){let e=matchMedia(t.value);e.matches?i():e.addEventListener(\"change\",i,{once:!0})}};(self.Astro||(self.Astro={})).media=n;window.dispatchEvent(new Event(\"astro:media\"));})();"],["only","(()=>{var e=async t=>{await(await t())()};(self.Astro||(self.Astro={})).only=e;window.dispatchEvent(new Event(\"astro:only\"));})();"],["visible","(()=>{var a=(s,i,o)=>{let r=async()=>{await(await s())()},t=typeof i.value==\"object\"?i.value:void 0,c={rootMargin:t==null?void 0:t.rootMargin},n=new IntersectionObserver(e=>{for(let l of e)if(l.isIntersecting){n.disconnect(),r();break}},c);for(let e of o.children)n.observe(e)};(self.Astro||(self.Astro={})).visible=a;window.dispatchEvent(new Event(\"astro:visible\"));})();"]],"entryModules":{"\u0000noop-middleware":"_noop-middleware.mjs","\u0000noop-actions":"_noop-actions.mjs","\u0000@astro-page:src/pages/about@_@astro":"pages/about.astro.mjs","\u0000@astro-page:src/pages/blog@_@astro":"pages/blog.astro.mjs","\u0000@astro-page:src/pages/contact@_@astro":"pages/contact.astro.mjs","\u0000@astro-page:src/pages/notes@_@astro":"pages/notes.astro.mjs","\u0000@astro-page:src/pages/notes/[...page]@_@astro":"pages/notes/_---page_.astro.mjs","\u0000@astro-page:src/pages/notes/[...slug]@_@astro":"pages/notes/_---slug_.astro.mjs","\u0000@astro-page:src/pages/posts/[slug]@_@astro":"pages/posts/_slug_.astro.mjs","\u0000@astro-page:src/pages/index@_@astro":"pages/index.astro.mjs","\u0000@astro-renderers":"renderers.mjs","\u0000@astrojs-manifest":"manifest_DfpgSt0u.mjs","/home/ancheng/WebApp/myblog/.astro/content-assets.mjs":"chunks/content-assets_DleWbedO.mjs","/home/ancheng/WebApp/myblog/.astro/content-modules.mjs":"chunks/content-modules_Dz-S_Wwv.mjs","\u0000astro:data-layer-content":"chunks/_astro_data-layer-content_BTZYPnox.mjs","/home/ancheng/WebApp/myblog/node_modules/astro/dist/assets/services/sharp.js":"chunks/sharp_BBUc8yMd.mjs","/home/ancheng/WebApp/myblog/src/pages/notes/[...slug].astro?astro&type=script&index=0&lang.ts":"_astro/_...slug_.astro_astro_type_script_index_0_lang.BNc1jUbE.js","/home/ancheng/WebApp/myblog/src/layouts/BlogPost.astro?astro&type=script&index=0&lang.ts":"_astro/BlogPost.astro_astro_type_script_index_0_lang.D6KJQTaM.js","/home/ancheng/WebApp/myblog/src/layouts/BlogPost.astro?astro&type=script&index=1&lang.ts":"_astro/BlogPost.astro_astro_type_script_index_1_lang.BNc1jUbE.js","/home/ancheng/WebApp/myblog/src/layouts/BlogPost.astro?astro&type=script&index=2&lang.ts":"_astro/BlogPost.astro_astro_type_script_index_2_lang.CfZGkvNd.js","/home/ancheng/WebApp/myblog/src/layouts/BlogPost.astro?astro&type=script&index=3&lang.ts":"_astro/BlogPost.astro_astro_type_script_index_3_lang.7sIC7wou.js","/home/ancheng/WebApp/myblog/src/layouts/BlogPost.astro?astro&type=script&index=4&lang.ts":"_astro/BlogPost.astro_astro_type_script_index_4_lang.CDpsKtqO.js","/home/ancheng/WebApp/myblog/src/components/ThemeToggle.astro?astro&type=script&index=0&lang.ts":"_astro/ThemeToggle.astro_astro_type_script_index_0_lang.BWcc2_Z1.js","astro:scripts/before-hydration.js":""},"inlinedScripts":[["/home/ancheng/WebApp/myblog/src/pages/notes/[...slug].astro?astro&type=script&index=0&lang.ts","document.addEventListener(\"DOMContentLoaded\",()=>{const e=document.getElementById(\"buttons-panel\");e?(e.classList.add(\"fixed\"),console.log(\"Class 'fixed' added to the buttons-panel element.\")):console.error(\"Element with ID 'buttons-panel' not found.\")});"],["/home/ancheng/WebApp/myblog/src/layouts/BlogPost.astro?astro&type=script&index=0&lang.ts","const e=document.getElementById(\"to-top-button\"),n=document.querySelector(\"header\");function c(t){t.forEach(o=>{e.dataset.show=(!o.isIntersecting).toString()})}e.addEventListener(\"click\",()=>{document.documentElement.scrollTo({behavior:\"smooth\",top:0})});const r=new IntersectionObserver(c);r.observe(n);"],["/home/ancheng/WebApp/myblog/src/layouts/BlogPost.astro?astro&type=script&index=1&lang.ts","document.addEventListener(\"DOMContentLoaded\",()=>{const e=document.getElementById(\"buttons-panel\");e?(e.classList.add(\"fixed\"),console.log(\"Class 'fixed' added to the buttons-panel element.\")):console.error(\"Element with ID 'buttons-panel' not found.\")});"],["/home/ancheng/WebApp/myblog/src/layouts/BlogPost.astro?astro&type=script&index=2&lang.ts","const t=document.getElementById(\"toggle-toc\"),n=document.getElementById(\"close-toc\"),e=document.getElementById(\"toc-panel\"),c=document.getElementById(\"toggle-toc-mobile\");if(!e)throw console.error(\"Element toc-panel not found\"),new Error(\"toc-panel is required\");const d=()=>{const o=window.matchMedia(\"(min-width: 768px)\").matches;return o&&e.classList.contains(\"md:block\")||!o&&!e.classList.contains(\"hidden\")},s=()=>{e.classList.add(\"hidden\"),e.classList.remove(\"md:block\")},i=()=>{e.classList.remove(\"hidden\"),e.classList.add(\"md:block\")},l=()=>{d()?s():i()};t?t.addEventListener(\"click\",l):console.error(\"Element toggle-toc not found\");c?c.addEventListener(\"click\",l):console.error(\"Element toggle-toc-mobile not found\");n?n.addEventListener(\"click\",s):console.error(\"Element close-toc not found\");"],["/home/ancheng/WebApp/myblog/src/layouts/BlogPost.astro?astro&type=script&index=3&lang.ts","const l=document.getElementById(\"toggle-panel\"),t=document.getElementById(\"close-panel\"),e=document.getElementById(\"series-panel\"),s=document.getElementById(\"toggle-panel-mobile\");if(!e)throw console.error(\"Element series-panel not found\"),new Error(\"series-panel is required\");const i=()=>{const n=window.matchMedia(\"(min-width: 1024px)\").matches;return n&&e.classList.contains(\"lg:block\")||!n&&!e.classList.contains(\"hidden\")},o=()=>{e.classList.add(\"opacity-0\",\"-translate-x-full\"),setTimeout(()=>{e.classList.remove(\"block\",\"lg:block\"),e.classList.add(\"hidden\")},300)},a=()=>{e.classList.remove(\"hidden\"),e.classList.add(\"block\",\"lg:block\"),setTimeout(()=>{e.classList.remove(\"opacity-0\",\"-translate-x-full\")},10)},c=()=>{i()?o():a()};l?l.addEventListener(\"click\",c):console.error(\"Element toggle-panel not found\");s?s.addEventListener(\"click\",c):console.error(\"Element toggle-panel-mobile not found\");t?t.addEventListener(\"click\",o):console.error(\"Element close-panel not found\");"],["/home/ancheng/WebApp/myblog/src/layouts/BlogPost.astro?astro&type=script&index=4&lang.ts","document.addEventListener(\"DOMContentLoaded\",()=>{document.querySelectorAll(\"pre\").forEach(o=>{if(!o.querySelector(\".copy-code\")){const t=document.createElement(\"button\");t.className=\"absolute flex items-center justify-center bg-bgColor h-6 font-medium overflow-hidden rounded-md text-light hover:text-accent-two font-sans text-sm font-medium top-2 right-2\";const e=document.createElement(\"span\");e.innerText=\"Copy\",e.className=\"flex items-center block w-full h-full px-2 bg-[var(--code-title-bg)]\",t.appendChild(e),o.appendChild(t),t.addEventListener(\"click\",async()=>{const n=o.querySelector(\"code\")?.textContent;n&&(await navigator.clipboard.writeText(n),e.innerText=\"Copied!\",setTimeout(()=>{e.innerText=\"Copy\"},1500))})}})});"],["/home/ancheng/WebApp/myblog/src/components/ThemeToggle.astro?astro&type=script&index=0&lang.ts","function t(){return typeof document<\"u\"?document.documentElement.getAttribute(\"data-theme\")===\"dark\":!1}class c extends HTMLElement{connectedCallback(){const e=this.querySelector(\"button\");e.setAttribute(\"role\",\"switch\"),e.setAttribute(\"aria-checked\",String(t())),e.addEventListener(\"click\",()=>{let n=new CustomEvent(\"theme-change\",{detail:{theme:t()?\"light\":\"dark\"}});document.dispatchEvent(n),e.setAttribute(\"aria-checked\",String(t()))})}}customElements.define(\"theme-toggle\",c);"]],"assets":["/verblog/file:///home/ancheng/WebApp/myblog/dist/about/index.html","/verblog/file:///home/ancheng/WebApp/myblog/dist/blog/index.html","/verblog/file:///home/ancheng/WebApp/myblog/dist/contact/index.html","/verblog/file:///home/ancheng/WebApp/myblog/dist/notes/index.html","/verblog/file:///home/ancheng/WebApp/myblog/dist/index.html"],"buildFormat":"directory","checkOrigin":false,"serverIslandNameMap":[],"key":"lI77/g6cigUiuAmn+mXDFqsKPt9jE/hVW8rTUh0x0f0="});
if (manifest.sessionConfig) manifest.sessionConfig.driverModule = null;

export { manifest };
