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

  // GitHub pages deployment config.
  organizationName: 'chipsalliance',
  projectName: 'chisel',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'throw',

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
          {to: '/docs/introduction', label: 'Docs', position: 'left'},
          {to: '/community', label: 'Community', position: 'left'},
          {to: '/api', label: 'API', position: 'left'},
          {
            href: 'https://github.com/chipsalliance/chisel',
            label: 'GitHub',
            position: 'right',
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
                label: 'Introduction',
                to: '/docs/introduction',
              },
              {
                label: 'ScalaDoc',
                href: 'https://javadoc.io/doc/org.chipsalliance/chisel_2.13/latest/index.html'
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
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} ChipsAlliance. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: [
          // The Scala grammar extends the java one
          // prism requires manually loading java first
          'java',
          'scala',
          'verilog',
        ],
      },
    }),

  plugins: [
    [
      '@docusaurus/plugin-client-redirects',
      {
        redirects: [
        ],
      },
    ],
  ],
};


module.exports = config;

