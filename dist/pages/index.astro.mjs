import { c as createAstro, a as createComponent, r as renderComponent, b as renderTemplate, m as maybeRenderHead, d as addAttribute } from '../chunks/astro/server_Dv6w92OM.mjs';
import 'kleur/colors';
import { g as getCollection, $ as $$FormattedDate } from '../chunks/FormattedDate_sDK99PUl.mjs';
import { $ as $$Base, a as $$Icon } from '../chunks/Base_CjJWARIv.mjs';
export { renderers } from '../renderers.mjs';

const $$Astro = createAstro("https://ver0810.github.io");
const $$Index = createComponent(async ($$result, $$props, $$slots) => {
  const Astro2 = $$result.createAstro($$Astro, $$props, $$slots);
  Astro2.self = $$Index;
  const MAX_POSTS = 10;
  const allPosts = await getCollection("post");
  const allPostsByDate = allPosts.sort(
    (a, b) => new Date(b.data.publishDate).getTime() - new Date(a.data.publishDate).getTime()
  ).slice(0, MAX_POSTS);
  const MAX_NOTES = 2;
  const allNotes = await getCollection("note");
  const latestNotes = allNotes.sort(
    (a, b) => new Date(b.data.publishDate).getTime() - new Date(a.data.publishDate).getTime()
  ).slice(0, MAX_NOTES);
  return renderTemplate`${renderComponent($$result, "Base", $$Base, {}, { "default": async ($$result2) => renderTemplate` ${maybeRenderHead()}<section class="text-center relative top-1/2 transform -translate-y-1/2"> <div class="absolute top-0 left-1/2 md:top-[-15%] -ml-[50vw] min-h-screen w-screen pointer-events-none blur-3xl opacity-50 overflow-x-hidden"> <div class="absolute top-[10%] right-[-40%] w-[65%] h-full bg-gradient-to-b from-purple-300 via-blue-300 to-transparent rounded-full opacity-30 dark:opacity-10"></div> <div class="absolute top-[10%] left-[-40%] w-[65%] h-full bg-gradient-to-b from-blue-300 via-pink-300 to-transparent rounded-full opacity-30 dark:opacity-10"></div> <div class="absolute top-[-20%] left-[-50%] w-[85%] h-full bg-gradient-to-b from-indigo-300 via-orange-300 to-transparent rounded-full opacity-60 dark:opacity-10"></div> <div class="absolute top-[-20%] right-[-50%] w-[85%] h-full bg-gradient-to-b from-orange-300 via-indigo-300 to-transparent rounded-full opacity-60 dark:opacity-10"></div> </div> <section class="max-w-xl mx-auto relative flex items-center justify-center h-screen -mt-24"> <div class="w-full text-center"> <span class="title text-3xl bg-gradient-to-r from-accent-two/85 via-accent-one/85 to-accent-two/85 dark:from-accent-two dark:via-accent-one dark:to-accent-two bg-clip-text text-transparent">
Introducing Astro Citrus!
</span> <p class="mt-4 mb-4 text-lg font-medium">
Hi, I’m Ikun, a computer science student.<br> I will share my notes and thoughts here. If you are interested in my work, please check out my blog and notes.
</p> <div class="flex justify-center mb-4"> <!-- <SocialList /> --> <!-- <p>SocialList</p> --> </div> <div class="flex justify-center space-x-4 mt-4"> <a href="/verblog/blog/" class="relative flex items-center justify-center h-8 px-4 rounded-lg shadow-lg hover:brightness-110 transition-all bg-gradient-to-r from-accent-one to-accent-two"> <span class="text-bgColor font-semibold"> Read Blog </span> </a> <a href="https://github.com/ver0810" class="relative flex items-center justify-center h-8 px-4 rounded-lg shadow-lg bg-special-lighter hover:brightness-110 hover:bg-special-lightest"> ${renderComponent($$result2, "Icon", $$Icon, { "class": "w-4 h-4 mr-1", "name": "lucide:github" })} <span class="bg-clip-text text-transparent font-semibold bg-gradient-to-r from-accent-one to-accent-two">
Github
</span> </a> </div> </div> </section> <!-- Posts Section --> <section aria-label="Blog post list" class="mt-16"> <div class="mb-8"> <h2 class="title text-2xl md:text-3xl mb-2"> <a href="/verblog/blog/" class="text-accent-two hover:text-accent-one transition-colors">Posts</a> </h2> <p class="text-lighter">
共 ${allPostsByDate.length} 篇博客文章
</p> </div> <div class="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3"> ${allPostsByDate.map((post) => renderTemplate`<article class="group h-full flex flex-col rounded-lg bg-color-75 px-4 md:px-6 py-4 transition-all duration-200 hover:shadow-lg hover:bg-color-100 border border-color-50"> <div class="flex-grow"> <h3 class="text-lg font-semibold mb-2 leading-tight"> <a${addAttribute(`/verblog/posts/${post.slug}/`, "href")} class="citrus-link hover:text-accent-one transition-colors"> ${post.data.draft && renderTemplate`<span class="text-red-500 text-sm">(Draft) </span>`} ${post.data.title} </a> </h3> <div class="flex items-center text-sm text-lighter mb-3"> <svg class="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"> <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path> </svg> ${renderComponent($$result2, "FormattedDate", $$FormattedDate, { "date": post.data.publishDate, "dateTimeOptions": {
    year: "numeric",
    month: "short",
    day: "numeric"
  } })} </div> ${post.data.description && renderTemplate`<p class="text-sm text-lighter leading-relaxed line-clamp-3"> ${post.data.description} </p>`} </div> <div class="mt-4 pt-3 border-t border-color-50/50"> <a${addAttribute(`/verblog/posts/${post.slug}/`, "href")} class="text-sm font-medium text-accent-one hover:text-accent-two transition-colors flex items-center">
阅读全文
<svg class="w-4 h-4 ml-1 transition-transform group-hover:translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"> <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path> </svg> </a> </div> </article>`)} </div> </section> <!-- Notes Section --> ${latestNotes.length > 0 && renderTemplate`<section class="mt-16"> <div class="mb-8"> <h2 class="title text-2xl md:text-3xl mb-2"> <a href="/verblog/notes/" class="text-accent-two hover:text-accent-one transition-colors">Notes</a> </h2> <p class="text-lighter">
最新 ${latestNotes.length} 篇笔记
</p> </div> <div class="grid grid-cols-1 gap-6 md:grid-cols-2"> ${latestNotes.map((note) => {
    const body = note.body || "";
    const contentText = body.replace(/^---[\s\S]*?---/, "").replace(/[#*\`\[\]]/g, "").replace(/\n+/g, " ").trim();
    const preview = contentText.substring(0, 100) + (contentText.length > 100 ? "..." : "");
    return renderTemplate`<article class="group h-full flex flex-col rounded-lg bg-color-75 px-4 md:px-6 py-4 transition-all duration-200 hover:shadow-lg hover:bg-color-100 border border-color-50"> <div class="flex-grow"> <h3 class="text-lg font-semibold mb-2 leading-tight"> <a${addAttribute(`/verblog/notes/${note.slug}/`, "href")} class="citrus-link hover:text-accent-one transition-colors"> ${note.data.title} </a> </h3> <div class="flex items-center text-sm text-lighter mb-3"> <svg class="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"> <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path> </svg> ${renderComponent($$result2, "FormattedDate", $$FormattedDate, { "date": note.data.publishDate, "dateTimeOptions": {
      year: "numeric",
      month: "short",
      day: "numeric"
    } })} </div> <p class="text-sm text-lighter leading-relaxed line-clamp-3"> ${note.data.description || preview} </p> </div> <div class="mt-4 pt-3 border-t border-color-50/50"> <a${addAttribute(`/verblog/notes/${note.slug}/`, "href")} class="text-sm font-medium text-accent-one hover:text-accent-two transition-colors flex items-center">
阅读全文
<svg class="w-4 h-4 ml-1 transition-transform group-hover:translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"> <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path> </svg> </a> </div> </article>`;
  })} </div> </section>`} </section> ` })}`;
}, "/home/ancheng/WebApp/myblog/src/pages/index.astro", void 0);

const $$file = "/home/ancheng/WebApp/myblog/src/pages/index.astro";
const $$url = "/verblog";

const _page = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
  __proto__: null,
  default: $$Index,
  file: $$file,
  url: $$url
}, Symbol.toStringTag, { value: 'Module' }));

const page = () => _page;

export { page };
