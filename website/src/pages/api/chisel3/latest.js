// SPDX-License-Identifier: Apache-2.0
import React from 'react';
import Layout from '@theme/Layout';
import BrowserOnly from '@docusaurus/BrowserOnly';
import useBaseUrl from '@docusaurus/useBaseUrl';

// This is a manual redirect.
// We cannot use @docusaurus/plugin-client-redirects because it only redirects
// to pages known to Docusaurus while the generated ScalaDoc is just a blob of
// static HTML.
export default function Redirect() {
  return (
    <BrowserOnly fallback={<div>Loading...</div>}>
      {() => {
        window.location.replace(useBaseUrl('/api/latest'));
      }}
    </BrowserOnly>
  );
}
