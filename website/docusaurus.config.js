// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Chisel',
  tagline: 'Software-defined hardware',
  favicon: 'img/chisel-tool-icon.svg',

  // Set the production url of your site here
  url: 'https://www.chisel-lang.org',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // Github pages adds trailing slashes by default, we cannot leave this
  // undefined or the Algolia webcrawler won't work.
  // One side-effect of making it false is that markdown files that have the
  // same name as their parent directory or have the name `index.md` now map
  // to `<parent directory>.html`. This means that any links they have to other
  // markdown files in the same directory now need to include the directory
  // path. For example:
  //
  // docs
  // └── foo
  //     ├── foo.md (or index.md)
  //     ├── bar.md
  //     └── fizz.md
  //
  // This maps to the following .html:
  // docs
  // ├── foo.html
  // └── foo
  //     ├── bar.html
  //     └── fizz.html
  //
  // A link from bar.md to fizz.md is [link](fizz).
  // A link from foo.md to fizz.md is [link](foo/fizz).
  trailingSlash: false,

  // GitHub pages deployment config.
  organizationName: 'chipsalliance',
  projectName: 'chisel',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'throw',
  onBrokenAnchors: 'throw',

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Use function to avoid extra 'docs/' in relative path of doc
          editUrl: (params) => {
            return (
              'https://github.com/chipsalliance/chisel/tree/main/docs/src/' +
              params.docPath
            );
          },
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    ({
      image: 'img/chisel-tool.svg',
      navbar: {
        title: 'Chisel',
        logo: {
          alt: 'Chisel Logo',
          src: 'img/chisel-tool.svg',
        },
        items: [
          {to: '/docs', label: 'Docs', position: 'left'},
          {to: '/community', label: 'Community', position: 'left'},
          {to: '/api', label: 'API', position: 'left'},
          {to: '/blog', label: 'Blog', position: 'left'},
          {
            href: 'https://www.chipsalliance.org/',
            position: 'right',
            className: 'header-chipsalliance-link',
            'aria-label': 'ChipsAlliance link',
          },
          {
            href: 'https://github.com/chipsalliance/chisel',
            position: 'right',
            className: 'header-github-link',
            'aria-label': 'GitHub repository',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Docs',
                to: '/docs',
              },
              {
                label: 'ScalaDoc',
                // pathname is needed by Docusaurus to correctly handle
                // internal links in the static folder, see
                // https://docusaurus.io/docs/advanced/routing#escaping-from-spa-redirects
                to: 'pathname:///api/latest',
              }
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/chisel',
              },
              {
                label: 'Gitter',
                href: 'https://gitter.im/freechipsproject/chisel3',
              },
              {
                label: 'Twitter',
                href: 'https://twitter.com/chisel_lang',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/chipsalliance/chisel',
              },
              {
                label: 'Blog',
                to: '/blog',
              }
            ],
          },
        ],
        copyright: `Copyright © Chisel a Series of LF Projects, LLC. For web site terms of use, trademark policy and other project policies please see https://lfprojects.org.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: [
          'bash',
          // The Scala grammar extends the java one
          // prism requires manually loading java first
          'java',
          'scala',
          'verilog',
        ],
      },
      // Algolia search integration: https://docusaurus.io/docs/search
      algolia: {
        appId: 'TTFNIAC3CJ',
        apiKey: '6a4ec6ef68c7b32aca79adf7440604ca',
        indexName: 'chisel-lang',
        // When true, this adds facets which (with default settings) break search results
        // It is probably fixable, see: https://docusaurus.io/docs/search#contextual-search
        contextualSearch: false,
        searchPagePath: 'search',
        insights: true,
      },
    }),

  plugins: [
    [
      '@docusaurus/plugin-client-redirects',
      { // Clean up links including .html, /myPages.html -> /myPage
        fromExtensions: ['html', 'htm'],
        redirects: [
          { // Redirect old chisel3 docs landing page
            from: '/chisel3',
            to: '/docs',
          },
        ],
        createRedirects(to) {
          // Redirect from all /chisel3/docs/X links to /docs/X
          // This catches many broken links from old sbt-microsites website
          if (to.includes('/docs')) {
            const from1 = to.replace('/docs', '/chisel3/docs');
            // Include redirects for .html/.htm because fromExtensions does not
            // compose with these redirects automatically.
            const from2 = from1.concat(".html");
            const from3 = from1.concat(".htm");
            return [from1, from2, from3];
          }
          return undefined;
        },
      },
    ],
  ],
};


module.exports = config;
