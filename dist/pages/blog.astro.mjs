import { a as createComponent, r as renderComponent, b as renderTemplate, m as maybeRenderHead, d as addAttribute } from '../chunks/astro/server_Dv6w92OM.mjs';
import 'kleur/colors';
import { g as getCollection, $ as $$FormattedDate } from '../chunks/FormattedDate_sDK99PUl.mjs';
import { $ as $$Base } from '../chunks/Base_B9sy8j0i.mjs';
export { renderers } from '../renderers.mjs';

const $$Blog = createComponent(async ($$result, $$props, $$slots) => {
  const tags = [
    "Test",
    "Test",
    "Test",
    "Test",
    "Test",
    "Test",
    "Test",
    "Test",
    "Test",
    "Test",
    "Test",
    "Test",
    "Test",
    "Test",
    "Test",
    "Test",
    "Test",
    "Test",
    "Test",
    "Test"
  ];
  const posts = await getCollection("post");
  const allPostsByDate = posts.sort(
    (a, b) => new Date(b.data.publishDate).getTime() - new Date(a.data.publishDate).getTime()
  );
  console.log(posts);
  return renderTemplate`${renderComponent($$result, "Base", $$Base, {}, { "default": async ($$result2) => renderTemplate` ${maybeRenderHead()}<div class="mb-8"> <h1 class="title text-2xl md:text-3xl mb-2">Posts</h1> <p class="text-lighter">
共 ${allPostsByDate.length} 篇博客文章
</p> </div> <div class="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-8"> <!-- Posts Container --> <section class="lg:col-span-1"> <div class="space-y-6"> ${allPostsByDate.map((post) => renderTemplate`<article class="group rounded-lg bg-color-75 px-6 py-4 transition-all duration-200 hover:shadow-lg hover:bg-color-100 border border-color-50"> <div class="flex items-start justify-between mb-3"> <h3 class="text-lg font-semibold leading-tight flex-1"> <a${addAttribute(`/verblog/posts/${post.slug}/`, "href")} class="citrus-link hover:text-accent-one transition-colors"> ${post.data.title} </a> </h3> ${renderComponent($$result2, "FormattedDate", $$FormattedDate, { "class": "text-sm text-lighter ml-4 flex-shrink-0", "date": post.data.publishDate, "dateTimeOptions": {
    year: "numeric",
    month: "short",
    day: "numeric"
  } })} </div> ${post.data.description && renderTemplate`<p class="text-sm text-lighter leading-relaxed line-clamp-2 mb-3"> ${post.data.description} </p>`} <div class="flex items-center justify-between"> <a${addAttribute(`/verblog/posts/${post.slug}/`, "href")} class="text-sm font-medium text-accent-one hover:text-accent-two transition-colors flex items-center">
阅读全文
<svg class="w-4 h-4 ml-1 transition-transform group-hover:translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"> <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path> </svg> </a> </div> </article>`)} </div> </section> <!-- Tags Container --> <aside class="lg:w-40"> <div class="sticky top-8"> <div class="rounded-lg bg-color-75 p-2 border border-color-50"> <h4 class="title mb-2 text-sm font-semibold text-accent-two">标签</h4> <div class="flex flex-wrap gap-1"> ${tags.map((tag) => renderTemplate`<a class="inline-block"> <div class="inline-flex items-center rounded px-1.5 py-0.5 text-xs font-medium transition-all duration-200 bg-color-100 text-textColor hover:bg-accent-one hover:text-bgColor"> <span class="mr-0.5">#</span> <span>Test</span> </div> </a>`)} </div> </div> </div> </aside> </div> <nav class="flex items-center gap-x-4 font-medium text-accent justify-end"> <a> Next Page --> </a> </nav> ` })}`;
}, "/home/ancheng/WebApp/myblog/src/pages/blog.astro", void 0);

const $$file = "/home/ancheng/WebApp/myblog/src/pages/blog.astro";
const $$url = "/verblog/blog";

const _page = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
  __proto__: null,
  default: $$Blog,
  file: $$file,
  url: $$url
}, Symbol.toStringTag, { value: 'Module' }));

const page = () => _page;

export { page };
