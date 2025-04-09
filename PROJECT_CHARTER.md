# Chisel Technical Charter  
### Adopted March 31, 2025

This Charter sets forth the responsibilities and procedures for technical contribution to, and oversight of, the Chisel open source project, which has been established as Chisel a Series of LF Projects, LLC (the “Project”). LF Projects, LLC (“LF Projects”) is a Delaware series limited liability company. All contributors (including committers, maintainers, and other technical positions) and other participants in the Project (collectively, “Collaborators”) must comply with the terms of this Charter.

---

## 1. Mission and Scope of the Project

a. The mission of the Project is to develop a stable, user-friendly platform for production-quality RTL generator development, designed to support the needs of a variety of RTL development communities, including industry, academia, and individuals.

b. The scope of the Project includes collaborative development under the Project License (as defined herein) supporting the mission, including documentation, testing, integration and the creation of other artifacts that aid the development, deployment, operation or adoption of the open source project.

---

## 2. Technical Steering Committee

a. The Technical Steering Committee (the “TSC”) will be responsible for all technical oversight of the open source Project.

b. The TSC voting members are initially the Project’s Committers. At the inception of the project, a list of voting members will be as set forth within the “CONTRIBUTING file” within the Project’s code repository. The TSC may choose an alternative approach for determining the voting members of the TSC, and any such alternative approach will be documented in the CONTRIBUTING file. Any meetings of the Technical Steering Committee are intended to be open to the public, and can be conducted electronically, via teleconference, or in person.

c. TSC projects generally will involve Contributors and Committers. The TSC may adopt or modify roles so long as the roles are documented in the CONTRIBUTING file. Unless otherwise documented:

  - i. Contributors include anyone in the technical community that contributes code, documentation, or other technical artifacts to the Project;
  - ii.Committers are Contributors who have earned the ability to modify (“commit”) source code, documentation or other technical artifacts in a project’s repository;
  - iii. A Contributor may become a Committer by a majority approval of the existing Committers. A Committer may be removed by a majority approval of the other existing Committers.

d. Participation in the Project through becoming a Contributor and Committer is open to anyone so long as they abide by the terms of this Charter.

e. The TSC may:
1. Establish work flow procedures for the submission, approval, and closure/archiving of projects,
2. Set requirements for the promotion of Contributors to Committer status, as applicable, and
3. Amend, adjust, refine and/or eliminate the roles of Contributors, and Committers, and create new roles, and publicly document any TSC roles, as it sees fit.

f. The TSC may elect a TSC Chair, who will preside over meetings of the TSC and will serve until their resignation or replacement by the TSC. The TSC Chair, or any other TSC member so designated by the TSC, will serve as the primary communication contact between the Project and CHIPS Alliance, a directed fund of The Linux Foundation.

g. Responsibilities. The TSC will be responsible for all aspects of oversight relating to the Project, which may include:

- i. Coordinating the technical direction of the Project;
- ii. Approving project or system proposals (including, but not limited to, incubation, deprecation, and changes to a sub-project’s scope);
- iii. Organizing sub-projects and removing sub-projects;
- iv. Creating sub-committees or working groups to focus on cross-project technical issues and requirements;
- v. Appointing representatives to work with other open source or open standards communities;
- vi. Establishing community norms, workflows, issuing releases, and security issue reporting policies;
- vii. Approving and implementing policies and processes for contributing (to be published in the CONTRIBUTING file) and coordinating with the series manager of the Project (as provided for in the Series Agreement, the “Series Manager”) to resolve matters or concerns that may arise as set forth in Section 7 of this Charter;
- viii. Discussions, seeking consensus, and where necessary, voting on technical matters relating to the code base that affect multiple projects;
- ix. Coordinating any marketing, events, or communications regarding the Project.

---

3. TSC Voting

- a. While the Project aims to operate as a consensus-based community, if any TSC decision requires a vote to move the Project forward, the voting members of the TSC will vote on a one vote per voting member basis.
- b. Quorum for TSC meetings requires at least fifty percent of all voting members of the TSC to be present. The TSC may continue to meet if quorum is not met but will be prevented from making any decisions at the meeting.
- c. Except as provided in Section 7.c. and 8.a, decisions by vote at a meeting require a majority vote of those in attendance, provided quorum is met. Decisions made by electronic vote without a meeting require a majority vote of all voting members of the TSC.
- d. In the event a vote cannot be resolved by the TSC, any voting member of the TSC may refer the matter to the Series Manager for assistance in reaching a resolution.

---

4. Compliance with Policies

- a. This Charter is subject to the Series Agreement for the Project and the Operating Agreement of LF Projects. Contributors will comply with the policies of LF Projects as may be adopted and amended by LF Projects, including, without limitation the policies listed at [https://lfprojects.org/policies](https://lfprojects.org/policies).
- b. The TSC may adopt a code of conduct (“CoC”) for the Project, which is subject to approval by the Series Manager. In the event that a Project-specific CoC has not been approved, the LF Projects Code of Conduct listed at [https://lfprojects.org/policies](https://lfprojects.org/policies) will apply for all Collaborators in the Project.
- c. When amending or adopting any policy applicable to the Project, LF Projects will publish such policy, as to be amended or adopted, on its website at least 30 days prior to such policy taking effect; provided, however, that in the case of any amendment of the Trademark Policy or Terms of Use of LF Projects, any such amendment is effective upon publication.
- d. All Collaborators must allow open participation from any individual or organization meeting the requirements for contributing under this Charter and any policies adopted for all Collaborators by the TSC, regardless of competitive interests.
- e. The Project will operate in a transparent, open, collaborative, and ethical manner at all times.

---

5. Community Assets

- a. LF Projects will hold title to all trade or service marks used by the Project (“Project Trademarks”), whether based on common law or registered rights. Project Trademarks will be transferred and assigned to LF Projects to hold on behalf of the Project.
- b. The Project will, as permitted and in accordance with such license from LF Projects, develop and own all Project GitHub and social media accounts, and domain name registrations created by the Project community.
- c. Under no circumstances will LF Projects be expected or required to undertake any action on behalf of the Project that is inconsistent with the tax-exempt status or purpose, as applicable, of the Joint Development Foundation or LF Projects, LLC.

---

## 6. General Rules and Operations

### a. The Project will:
- i. Engage in the work of the Project in a professional manner consistent with maintaining a cohesive community, while also maintaining the goodwill and esteem of LF Projects, Joint Development Foundation and other partner organizations in the open source community; and
- ii. Respect the rights of all trademark owners, including any branding and trademark usage guidelines.

---

7. Intellectual Property Policy

a. Collaborators acknowledge that the copyright in all new contributions will be retained by the copyright holder as independent works of authorship and that no contributor or copyright holder will be required to assign copyrights to the Project.

b. Except as described in Section 7.c., all contributions to the Project are subject to the following:

- i. All new inbound code contributions to the Project must be made using [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0) (the “Project License”).
- ii. All new inbound code contributions must also be accompanied by a [Developer Certificate of Origin](http://developercertificate.org) sign-off in the source code system that is submitted through a TSC-approved contribution process.
- iii. All outbound code will be made available under the Project License.
- iv. Documentation will be received and made available by the Project under the Project License.
- v. The Project may seek to integrate and contribute back to other open source projects (“Upstream Projects”), conforming to all license requirements of those projects.

c. The TSC may approve the use of an alternative license or licenses for inbound or outbound contributions on an exception basis. License exceptions must be approved by a two-thirds vote of the entire TSC.

d. Contributed files should contain license information, such as SPDX short form identifiers, indicating the open source license or licenses pertaining to the file.

---

8. Amendments

This Charter may be amended by a two-thirds vote of the entire TSC and is subject to approval by LF Projects.
