import { c as createAstro, a as createComponent, m as maybeRenderHead, d as addAttribute, r as renderComponent, b as renderTemplate, f as renderSlot, F as Fragment, e as renderScript } from '../../chunks/astro/server_Dv6w92OM.mjs';
import 'kleur/colors';
import { g as getCollection, $ as $$FormattedDate, r as renderEntry } from '../../chunks/FormattedDate_sDK99PUl.mjs';
import { a as $$Icon, $ as $$Base } from '../../chunks/Base_B9sy8j0i.mjs';
import '@astrojs/internal-helpers/path';
import '@astrojs/internal-helpers/remote';
import { $ as $$Image } from '../../chunks/_astro_assets_WLVmmWF4.mjs';
import 'clsx';
export { renderers } from '../../renderers.mjs';

function injectChild(items, item) {
  const lastItem = items.at(-1);
  if (!lastItem || lastItem.depth >= item.depth) {
    items.push(item);
  } else {
    injectChild(lastItem.children, item);
    return;
  }
}
function generateToc(headings, { maxHeadingLevel = 6, minHeadingLevel = 1 } = {}) {
  const bodyHeadings = headings.filter(
    ({ depth }) => depth >= minHeadingLevel && depth <= maxHeadingLevel
  );
  const toc = [];
  for (const heading of bodyHeadings) injectChild(toc, { ...heading, children: [] });
  return toc;
}

const $$Astro$6 = createAstro("https://ver0810.github.io");
const $$TOCHeading = createComponent(($$result, $$props, $$slots) => {
  const Astro2 = $$result.createAstro($$Astro$6, $$props, $$slots);
  Astro2.self = $$TOCHeading;
  const {
    heading: { children, slug, text }
  } = Astro2.props;
  return renderTemplate`${maybeRenderHead()}<li class=""> <a${addAttribute(`Scroll to section: ${text}`, "aria-label")} class="text-light mt-1 line-clamp-2 break-words [padding-left:1ch] [text-indent:-1ch] before:text-accent-two before:content-['#'] hover:text-accent-two"${addAttribute(`#${slug}`, "href")}>${text} </a> ${!!children.length && renderTemplate`<ul class="ms-2"> ${children.map((subheading) => renderTemplate`${renderComponent($$result, "Astro.self", Astro2.self, { "heading": subheading })}`)} </ul>`} </li>`;
}, "/home/ancheng/WebApp/myblog/src/components/blog/TOCHeading.astro", void 0);

const $$Astro$5 = createAstro("https://ver0810.github.io");
const $$TOC = createComponent(($$result, $$props, $$slots) => {
  const Astro2 = $$result.createAstro($$Astro$5, $$props, $$slots);
  Astro2.self = $$TOC;
  const { headings } = Astro2.props;
  const toc = generateToc(headings);
  return renderTemplate`${maybeRenderHead()}<div class="sticky top-20 rounded-t-lg"> <div class="sticky top-20 bg-bgColor rounded-t-lg"> <div class="sticky top-20 flex pt-4 ps-8 pb-2 items-end title rounded-t-lg bg-color-75 pe-4 gap-x-1 border-t border-l border-r border-special-light"> <!--
			<Icon aria-hidden="true" class="flex-shrink-0 h-8 w-6 py-1" focusable="false" name="solar:clipboard-list-line-duotone" />
			--> <h4 class="title">Table of Contents</h4> <button id="close-toc" class="absolute top-4 right-4 h-8 w-8 flex items-center justify-center rounded-lg bg-accent-base/5 text-accent-base hover:bg-accent-base/10" aria-label="Close TOC"> ${renderComponent($$result, "Icon", $$Icon, { "aria-hidden": "true", "class": "h-6 w-6", "focusable": "false", "name": "fa-solid:times" })} </button> </div> </div> <div class="bg-bgColor rounded-b-lg"> <div class="rounded-b-lg pb-6 bg-color-75 border-b border-l border-r border-special-light"> <div class="max-h-[calc(100vh-11rem)] h-auto overflow-y-auto overflow-hidden px-8"> <ul class="text-sm font-medium text-textColor"> ${toc.map((heading) => renderTemplate`${renderComponent($$result, "TOCHeading", $$TOCHeading, { "heading": heading })}`)} </ul> </div> </div> </div> </div>`;
}, "/home/ancheng/WebApp/myblog/src/components/blog/TOC.astro", void 0);

const $$Astro$4 = createAstro("https://ver0810.github.io");
const $$Badge = createComponent(($$result, $$props, $$slots) => {
  const Astro2 = $$result.createAstro($$Astro$4, $$props, $$slots);
  Astro2.self = $$Badge;
  const { variant = "default", showHash = true, title } = Astro2.props;
  const badgeClasses = {
    base: "flex items-baseline pt-[0.075rem] drop-shadow-lg active:drop-shadow-none rounded-lg h-6 px-2 text-sm font-medium transition-colors",
    variants: {
      default: "bg-textColor text-bgColor hover:brightness-105",
      accent: "bg-accent text-bgColor hover:brightness-105",
      "accent-base": "bg-accent-base text-bgColor hover:brightness-105",
      "accent-one": "bg-accent-one text-bgColor hover:brightness-105",
      "accent-two": "bg-accent-two text-bgColor hover:brightness-105",
      muted: "bg-color-100 text-textColor hover:bg-accent-two hover:text-bgColor drop-shadow-none hover:drop-shadow-lg",
      outline: "border border-lightest text-textColor drop-shadow-none",
      inactive: "text-lighter bg-color-100 drop-shadow-none"
    }
  };
  const variantClasses = badgeClasses.variants[variant];
  return renderTemplate`${maybeRenderHead()}<div${addAttribute(`${badgeClasses.base} ${variantClasses}`, "class")}> ${showHash && renderTemplate`<span class="h-full">#</span>`} ${title} ${renderSlot($$result, $$slots["default"])} </div>`;
}, "/home/ancheng/WebApp/myblog/src/components/Badge.astro", void 0);

const $$Astro$3 = createAstro("https://ver0810.github.io");
const $$Separator = createComponent(($$result, $$props, $$slots) => {
  const Astro2 = $$result.createAstro($$Astro$3, $$props, $$slots);
  Astro2.self = $$Separator;
  const { type = "horizontal", className = "" } = Astro2.props;
  const separatorClasses = {
    base: "flex-shrink-0 bg-lighter mx-2",
    // Общие стили
    types: {
      horizontal: "h-[1px] w-full",
      // Горизонтальный разделитель
      vertical: "h-full w-[1px]",
      // Вертикальный разделитель
      dot: "w-1.5 h-1.5 rounded-full"
      // Кружок для dot
    }
    // Указываем, что типы здесь конкретные строки
  };
  const typeClass = separatorClasses.types[type];
  return renderTemplate`${type === "dot" ? renderTemplate`${maybeRenderHead()}<span${addAttribute(`${separatorClasses.base} ${typeClass} ${className}`, "class")}></span>` : renderTemplate`<span role="separator"${addAttribute(type === "horizontal" ? "horizontal" : "vertical", "aria-orientation")}${addAttribute(`${separatorClasses.base} ${typeClass} ${className}`, "class")}></span>`}`;
}, "/home/ancheng/WebApp/myblog/src/components/Separator.astro", void 0);

const $$Astro$2 = createAstro("https://ver0810.github.io");
const $$Masthead = createComponent(async ($$result, $$props, $$slots) => {
  const Astro2 = $$result.createAstro($$Astro$2, $$props, $$slots);
  Astro2.self = $$Masthead;
  const {
    content
  } = Astro2.props;
  const dateTimeOptions = {
    month: "long"
  };
  const postSeries = content.data.seriesId ? await getCollection("series").then((series) => series.find((s) => s.id === content.data.seriesId)).catch((err) => {
    console.error("Failed to find series:", err);
    return null;
  }) : null;
  return renderTemplate`${maybeRenderHead()}<div class="md:sticky md:top-8 md:z-10 flex items-end"> ${postSeries ? renderTemplate`<button id="toggle-panel" class="hidden md:flex mr-2 h-8 w-8 items-center bg-accent-base/10 flex-shrink-0 justify-center rounded-lg text-accent-base hover:brightness-110" aria-label="Toggle Series Panel" aria-controls="series-panel">  ${renderComponent($$result, "Icon", $$Icon, { "aria-hidden": "true", "class": "flex-shrink-0 h-6 w-6", "focusable": "false", "name": "solar:notes-bold" })} </button>` : null} ${!!content.rendered?.metadata?.headings?.length && renderTemplate`<button id="toggle-toc" class="hidden md:flex h-8 w-8 items-center flex-shrink-0 bg-accent-base/10 justify-center rounded-lg text-accent-base hover:brightness-110" aria-label="Table of Contents"> ${renderComponent($$result, "Icon", $$Icon, { "aria-hidden": "true", "class": "h-6 w-6", "focusable": "false", "name": "solar:clipboard-list-bold" })} </button>`} <h1 class="title ml-2 md:sticky md:top-4 md:z-20 line-clamp-none md:line-clamp-1 md:max-w-[calc(100%-10rem)]"${addAttribute(content.data.title, "title")} data-pagefind-body> ${content.data.title} </h1> </div> <div class="flex flex-wrap items-center text-lighter text-sm mt-[1.0625rem] mx-2 mb-2"> <span class="flex items-center h-[1.75rem]"> ${renderComponent($$result, "Icon", $$Icon, { "aria-hidden": "true", "class": "flex items-center h-4 w-4 me-1", "focusable": "false", "name": "hugeicons:calendar-03" })} ${renderComponent($$result, "FormattedDate", $$FormattedDate, { "date": content.data.publishDate, "dateTimeOptions": dateTimeOptions, "class": "flex flex-shrink-0" })} </span> ${renderComponent($$result, "Separator", $$Separator, { "type": "dot" })} <span class="flex items-center h-[1.75rem]"> ${renderComponent($$result, "Icon", $$Icon, { "aria-hidden": "true", "class": "flex items-center inline-block h-4 w-4 me-1", "focusable": "false", "name": "hugeicons:book-open-01" })}  ${content.rendered?.metadata?.frontmatter?.readingTime ? `${content.rendered.metadata.frontmatter.readingTime}` : "Less than one minute read"} </span> ${content.data.updatedDate && renderTemplate`${renderComponent($$result, "Separator", $$Separator, { "type": "dot" })}
			<span class="h-[1.75rem] flex items-center flex-shrink-0 rounded-lg bg-accent-two/5 text-accent-two py-1 px-2 text-sm gap-x-1">
Updated:${renderComponent($$result, "FormattedDate", $$FormattedDate, { "class": "flex flex-shrink-0", "date": content.data.updatedDate, "dateTimeOptions": dateTimeOptions })} </span>`} </div> ${content.data.draft ? renderTemplate`<span class="text-base text-red-500 ml-2">(Draft)</span>` : null} ${content.data.coverImage && renderTemplate`<div class="mb-4 mt-2 overflow-auto rounded-lg"> ${renderComponent($$result, "Image", $$Image, { "alt": content.data.coverImage.alt, "class": "object-cover", "fetchpriority": "high", "loading": "lazy", "loading": "eager", "src": content.data.coverImage.src })} </div>`} <p class="prose max-w-none text-textColor mx-2" data-pagefind-body> ${content.data.description} </p> <div class="mt-4 flex flex-wrap items-center gap-2 mx-1">  ${content.data.tags?.length ? renderTemplate`${renderComponent($$result, "Icon", $$Icon, { "aria-hidden": "true", "class": "flex-shrink-0 inline-block h-6 w-6 text-accent-base", "focusable": "false", "name": "solar:tag-line-duotone" })}	
			${renderComponent($$result, "Fragment", Fragment, {}, { "default": async ($$result2) => renderTemplate`${content.data.tags.map((tag) => renderTemplate`<a${addAttribute(`View all posts with the tag: ${tag}`, "aria-label")}${addAttribute(`/tags/${tag}`, "href")}> ${renderComponent($$result2, "Badge", $$Badge, { "variant": "accent-two", "title": tag })} </a>`)}` })}` : renderTemplate`${renderComponent($$result, "Icon", $$Icon, { "aria-hidden": "true", "class": "flex-shrink-0 inline-block h-6 w-6 text-lightest", "focusable": "false", "name": "solar:tag-line-duotone" })}
			<span class="text-sm text-lightest">No tags</span>`}  ${postSeries ? renderTemplate`<div class="flex items-center gap-2"> ${renderComponent($$result, "Icon", $$Icon, { "aria-hidden": "true", "class": "flex-shrink-0 inline-block h-6 w-6 text-accent-base", "focusable": "false", "name": "solar:notes-line-duotone" })} <a${addAttribute(`About ${postSeries.data.title} series`, "aria-label")}${addAttribute(`/series/${postSeries.id}`, "href")} class="flex items-center gap-2 flex-wrap"> ${renderComponent($$result, "Badge", $$Badge, { "variant": "accent-base", "showHash": false, "title": postSeries.data.title })} </a> </div>` : renderTemplate`<div class="flex items-center gap-2"> ${renderComponent($$result, "Icon", $$Icon, { "aria-hidden": "true", "class": "flex-shrink-0 inline-block h-6 w-6 text-lightest", "focusable": "false", "name": "solar:notes-line-duotone" })} <span class="text-sm text-lightest">Not in series</span> </div>`} </div>`;
}, "/home/ancheng/WebApp/myblog/src/components/blog/Masthead.astro", void 0);

const $$Astro$1 = createAstro("https://ver0810.github.io");
const $$BlogPost = createComponent(async ($$result, $$props, $$slots) => {
  const Astro2 = $$result.createAstro($$Astro$1, $$props, $$slots);
  Astro2.self = $$BlogPost;
  const { post } = Astro2.props;
  const { title, description, updatedDate, publishDate } = post.data;
  const articleDate = updatedDate?.toISOString() ?? publishDate.toISOString();
  const { headings } = await renderEntry(post);
  return renderTemplate`${renderComponent($$result, "BaseLayout", $$Base, { "meta": { articleDate, description, title } }, { "default": async ($$result2) => renderTemplate` ${renderComponent($$result2, "Masthead", $$Masthead, { "content": post })} ${maybeRenderHead()}<div class="mt-6 flex sm:grid-cols-[auto_1fr] md:items-start gap-x-8"> <article class="grid flex-grow grid-cols-1 break-words pt-4" data-pagefind-body> <div class="prose prose-neutral max-w-none flex-grow prose-headings:font-semibold prose-headings:text-accent-base prose-headings:before:text-accent-two sm:prose-headings:before:content-['#'] sm:prose-th:before:content-none"> ${renderSlot($$result2, $$slots["default"])} </div> </article> ${!!headings.length && renderTemplate`<aside id="toc-panel" class="md:sticky md:top-20 z-10 hidden md:w-[14rem] md:min-w-[14rem] md:rounded-lg md:block"> ${renderComponent($$result2, "TOC", $$TOC, { "headings": headings })} </aside>`} </div> <div class="left-0 right-12 z-50 ml-auto w-fit md:absolute"> <button id="to-top-button" class="fixed bottom-14 flex h-12 w-12 text-light translate-y-28 items-center justify-center rounded-full bg-bgColor text-3xl drop-shadow-xl transition-all duration-300 hover:text-accent-two data-[show=true]:translate-y-0 data-[show=true]:opacity-100" aria-label="Back to Top" data-show="false"> <span class="absolute inset-0 rounded-full bg-special-lighter flex items-center justify-center" aria-hidden="true"> <svg class="h-6 w-6" fill="none" focusable="false" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"> <path d="M4.5 15.75l7.5-7.5 7.5 7.5" stroke-linecap="round" stroke-linejoin="round"></path> </svg> </span> </button> </div> ` })} <!-- Copy code button --> ${renderScript($$result, "/home/ancheng/WebApp/myblog/src/layouts/BlogPost.astro?astro&type=script&index=0&lang.ts")} ${renderScript($$result, "/home/ancheng/WebApp/myblog/src/layouts/BlogPost.astro?astro&type=script&index=1&lang.ts")} ${renderScript($$result, "/home/ancheng/WebApp/myblog/src/layouts/BlogPost.astro?astro&type=script&index=2&lang.ts")} ${renderScript($$result, "/home/ancheng/WebApp/myblog/src/layouts/BlogPost.astro?astro&type=script&index=3&lang.ts")} <!-- Scroll to top button --> ${renderScript($$result, "/home/ancheng/WebApp/myblog/src/layouts/BlogPost.astro?astro&type=script&index=4&lang.ts")}`;
}, "/home/ancheng/WebApp/myblog/src/layouts/BlogPost.astro", void 0);

const $$Astro = createAstro("https://ver0810.github.io");
async function getStaticPaths() {
  const posts = await getCollection("post");
  return posts.map((post) => ({
    params: { slug: post.slug },
    props: { post }
  }));
}
const $$slug = createComponent(async ($$result, $$props, $$slots) => {
  const Astro2 = $$result.createAstro($$Astro, $$props, $$slots);
  Astro2.self = $$slug;
  const { post } = Astro2.props;
  if (!post) {
    return new Response(null, {
      status: 404,
      statusText: "Not found"
    });
  }
  const { Content } = await post.render();
  return renderTemplate`${renderComponent($$result, "PostLayout", $$BlogPost, { "post": post }, { "default": async ($$result2) => renderTemplate` ${renderComponent($$result2, "Content", Content, {})} ` })}`;
}, "/home/ancheng/WebApp/myblog/src/pages/posts/[slug].astro", void 0);

const $$file = "/home/ancheng/WebApp/myblog/src/pages/posts/[slug].astro";
const $$url = "/verblog/posts/[slug]";

const _page = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
	__proto__: null,
	default: $$slug,
	file: $$file,
	getStaticPaths,
	url: $$url
}, Symbol.toStringTag, { value: 'Module' }));

const page = () => _page;

export { page };
