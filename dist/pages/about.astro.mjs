import { c as createAstro, a as createComponent, r as renderComponent, b as renderTemplate, m as maybeRenderHead } from '../chunks/astro/server_Dv6w92OM.mjs';
import 'kleur/colors';
import { $ as $$Base } from '../chunks/Base_CjJWARIv.mjs';
export { renderers } from '../renderers.mjs';

const $$Astro = createAstro("https://ver0810.github.io");
const $$About = createComponent(($$result, $$props, $$slots) => {
  const Astro2 = $$result.createAstro($$Astro, $$props, $$slots);
  Astro2.self = $$About;
  return renderTemplate`${renderComponent($$result, "Base", $$Base, {}, { "default": ($$result2) => renderTemplate` ${maybeRenderHead()}<div class="max-w-2xl mx-auto relative flex h-screen -mt-24"> <div class=" w-full h-screen mt-10"> <h1 class="text-3xl font-semibold"># About</h1> <p>
This is the about page. It's a simple page with some text to show how
        Astro works.
</p> </div> </div> ` })}`;
}, "/home/ancheng/WebApp/myblog/src/pages/about.astro", void 0);

const $$file = "/home/ancheng/WebApp/myblog/src/pages/about.astro";
const $$url = "/verblog/about";

const _page = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
  __proto__: null,
  default: $$About,
  file: $$file,
  url: $$url
}, Symbol.toStringTag, { value: 'Module' }));

const page = () => _page;

export { page };
