import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Scala',
    Svg: require('@site/static/img/scala.svg').default,
    description: (
      <>
        Chisel is powered by Scala and brings all the power of object-oriented and
        functional programming to type-safe hardware design and generation.
      </>
    ),
  },
  {
    title: 'Chisel',
    Svg: require('@site/static/img/chisel-tool.svg').default,
    description: (
      <>
        Chisel, the Chisel standard library, and Chisel testing infrastructure
        enable agile, expressive, and reusable hardware design methodologies.
      </>
    ),
  },
  {
    title: 'FIRRTL',
    Svg: require('@site/static/img/firrtl_logo.svg').default,
    description: (
      <>
        The FIRRTL circuit compiler starts after Chisel and enables backend
        (FPGA, ASIC, technology) specialization, automated circuit transformation,
        and Verilog generation.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
