import { a as createComponent, r as renderComponent, b as renderTemplate, m as maybeRenderHead, d as addAttribute } from '../chunks/astro/server_Dv6w92OM.mjs';
import 'kleur/colors';
import { g as getCollection, $ as $$FormattedDate } from '../chunks/FormattedDate_sDK99PUl.mjs';
import { $ as $$Base } from '../chunks/Base_CjJWARIv.mjs';
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
</p> </div> <div class="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3"> ${allPostsByDate.map((post) => renderTemplate`<article class="group h-full flex flex-col rounded-lg bg-color-75 px-4 md:px-6 py-4 transition-all duration-200 hover:shadow-lg hover:bg-color-100 border border-color-50"> <div class="flex-grow"> <h3 class="text-lg font-semibold mb-2 leading-tight"> <a${addAttribute(`/verblog/posts/${post.slug}/`, "href")} class="citrus-link hover:text-accent-one transition-colors"> ${post.data.title} </a> </h3> <div class="flex items-center text-sm text-lighter mb-3"> <svg class="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"> <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path> </svg> ${renderComponent($$result2, "FormattedDate", $$FormattedDate, { "date": post.data.publishDate, "dateTimeOptions": {
    year: "numeric",
    month: "short",
    day: "numeric"
  } })} </div> ${post.data.description && renderTemplate`<p class="text-sm text-lighter leading-relaxed line-clamp-3"> ${post.data.description} </p>`} </div> <div class="mt-4 pt-3 border-t border-color-50/50"> <a${addAttribute(`/verblog/posts/${post.slug}/`, "href")} class="text-sm font-medium text-accent-one hover:text-accent-two transition-colors flex items-center">
阅读全文
<svg class="w-4 h-4 ml-1 transition-transform group-hover:translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"> <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path> </svg> </a> </div> </article>`)} </div> <aside class="md:min-w-[14rem] md:max-w-[14rem] md:basis-[14rem]"> <div class="sticky top-8"> <h4 class="title mb-4 flex gap-2 text-lg">标签</h4> <div class="space-y-2"> ${tags.map((tag) => renderTemplate`<a class="block"> <div class="flex items-center rounded-lg px-3 py-2 text-sm font-medium transition-all duration-200 bg-color-75 text-textColor hover:bg-color-100 border border-color-50 hover:border-color-100"> <span class="mr-2">#</span>
Test
</div> </a>`)} </div> </div> </aside> ` })}`;
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
