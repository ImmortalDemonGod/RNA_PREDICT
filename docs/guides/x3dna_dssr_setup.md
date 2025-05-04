# Setting Up X3DNA-DSSR for RNA Structure Analysis

## Overview

This guide outlines the process for obtaining and setting up X3DNA-DSSR, which is required for generating ground truth torsion angle data for our RNA prediction pipeline.

## Step 1: Obtain X3DNA-DSSR License and Binary

1. **Request an Academic License**:
   - Visit the [Columbia Technology Ventures DSSR page](https://inventions.techventures.columbia.edu/technologies/dssr-an-integrated--CU20391)
   - Click on "Express Licensing"
   - Sign in or create an account
   - Request the "DSSR-Basic, Academic (1 seat)" license, which is now free for academic users thanks to NIH funding
   - Complete the license request form with your academic credentials
   - Wait for approval (usually within 1-2 business days)

2. **Download the Binary**:
   - After your license is approved, you'll receive an email with download instructions
   - Download the appropriate binary for your operating system (Linux, macOS, or Windows)
   - The current version as of this writing is DSSR v2.5.2-2025apr03

## Step 2: Install X3DNA-DSSR

### For macOS/Linux:

1. Create a directory for the binary:
   ```bash
   mkdir -p ~/bin/x3dna-dssr
   ```

2. Move the downloaded binary to this directory:
   ```bash
   mv /path/to/downloaded/x3dna-dssr ~/bin/x3dna-dssr/
   ```

3. Make the binary executable:
   ```bash
   chmod +x ~/bin/x3dna-dssr/x3dna-dssr
   ```

4. Add to your PATH by adding the following line to your `~/.bashrc`, `~/.zshrc`, or equivalent shell configuration file:
   ```bash
   export PATH="$HOME/bin/x3dna-dssr:$PATH"
   ```

5. Apply the changes:
   ```bash
   source ~/.bashrc  # or source ~/.zshrc
   ```

### For Windows:

1. Create a folder for the binary, e.g., `C:\Program Files\x3dna-dssr`
2. Move the downloaded binary to this folder
3. Add the folder to your system PATH:
   - Right-click on "This PC" or "My Computer" and select "Properties"
   - Click on "Advanced system settings"
   - Click on "Environment Variables"
   - Under "System variables", find and select "Path", then click "Edit"
   - Click "New" and add the path to the folder (e.g., `C:\Program Files\x3dna-dssr`)
   - Click "OK" on all dialogs to save the changes

## Step 3: Verify Installation

Test that X3DNA-DSSR is properly installed and accessible:

```bash
x3dna-dssr --version
```

You should see the version information displayed.

## Next Steps

Once X3DNA-DSSR is installed and working, we'll proceed with:

1. Creating a script to automate running DSSR over our dataset structure files
2. Saving the results as `.pt` files containing the radian angle tensors
3. Integrating these ground truth angles into our training pipeline

## References

- [X3DNA-DSSR Official Website](http://home.x3dna.org/)
- [DSSR Documentation](http://docs.x3dna.org/dssr-manual.pdf)
- [DSSR Paper](https://doi.org/10.1093/nar/gkv716)

## License

DSSR: an integrated software tool for dissecting the spatial structure of RNA
Overview
Request Info
More Like This
Express Licensing
DSSR-Basic, Academic (1 seat), Now Free!

First name *
Last name *
Organization name *
Job title *
Address *
Apartment, suite, etc. (optional)
City *
Country *
State, territory, or province *
Postal code *
Telephone number *
Agreement
You must accept the terms of the agreement using the checkbox at the bottom of the page.

DSSR-Basic, Academic (1 seat), Now Free!
CU20391 -- DSSR: an integrated software tool for dissecting the spatial structure of RNA

You:
Tom Riddle
CU Biology
END USER LICENSE AGREEMENT

BY DOWNLOADING, INSTALLING OR USING THE ABOVE-NAMED SOFTWARE PROGRAM OR ITS RELATED DOCUMENTATION (COLLECTIVELY, THE "PROGRAM"), YOU ACKNOWLEDGE THAT YOU HAVE READ ALL OF THE TERMS AND CONDITIONS OF THIS AGREEMENT, UNDERSTAND THEM, AND AGREE TO BE BOUND BY THEM. WE RECOMMEND THAT YOU PRINT A COPY OF THIS AGREEMENT FOR YOUR RECORDS.

IF YOU DO NOT AGREE TO ALL OF THE TERMS OF THIS AGREEMENT, YOU MUST NOT DOWNLOAD, INSTALL OR USE THE PROGRAM.

YOU HEREBY REPRESENT AND WARRANT THAT YOU HAVE THE LEGAL AUTHORITY TO BIND THE ORGANIZATION NAMED IN YOUR REGISTRATION FORM, IF ANY, AND IF SUCH AN ORGANIZATION IS NAMED, SUCH ORGANIZATION SHALL BE DEEMED TO BE "YOU" FOR THE PURPOSE OF THIS AGREEMENT. IF NO SUCH ORGANIZATION IS NAMED, THEN "YOU" SHALL REFER TO YOU INDIVIDUALLY.

This Software License Agreement (the "Agreement") is between The Trustees of Columbia University in the City of New York, a non-profit private educational institution, having a principal place of business at 116th St. and Broadway, New York, New York 10027, U.S.A. ("Columbia") and You (as defined above).

1. License Grant. Under Columbia’s rights, Columbia grants You a non-exclusive and non-transferable license to install, display, and use one (1) copy of the Program. Columbia reserves the right to make corrections, improvements, or enhancements to the Program without notice to You and without obligation to furnish the said corrections, improvements, or enhancements to You.

2. Restrictions. You will not (i) reproduce or copy the Program, except that You may make one (1) copy of the Program solely for archival purposes, provided that You agree to reproduce all copyright and other proprietary right notices on the archival copy; (ii) use, or cause or permit the use of, the Program in whole or in part for any purpose other than as permitted under this Agreement; (iii) distribute, sell, lease, sublicense or otherwise transfer rights to the Program to any third party; (iv) reverse engineer, decompile, disassemble or otherwise attempt to derive the source code for the Program (except to the extent applicable laws specifically prohibit such restriction); (v) modify or create any derivative works of the Program, including translation or localization; or (vi) remove or alter any patent, trademark, logo, copyright or other proprietary notices, legends, symbols or labels in the Program.

3. License Fee. Now free, as

4. Term and Termination. The term of this Agreement shall continue until terminated in accordance with this Section 4. You may terminate this Agreement at any time by destroying all copies of the Program. This Agreement and the rights granted under this Agreement will terminate automatically, and without any further notice from or action by Columbia, if You fail to comply with any obligation set forth herein. Upon termination, You must immediately cease use and destroy all copies of the Program and verify such destruction in writing. Columbia shall have the right to disable electronically Your unauthorized use of the Program and resort to other "self-help" measures Columbia deems appropriate. Sections 2, 4-10, and 12-14 shall survive expiration or termination of this Agreement.

5. No Obligation to Support. It is understood and agreed that Columbia will provide no maintenance or installation services of any kind, error corrections, bug fixes, patches, updates or other modifications hereunder. In the event that Columbia, at its sole option, provides updates, error corrections, bug fixes, patches or other modifications to the Program to You ("Program Updates"), the Program Updates will be considered part of the Program, and subject to the terms and conditions of this Agreement.

6. Proprietary Rights. Title to the Program, and patents, copyrights, trademarks, and all other intellectual property rights applicable thereto, shall at all times remain solely and exclusively with Columbia and its suppliers, and You shall not take any action inconsistent with such ownership. Any rights not expressly granted herein are reserved to Columbia and its suppliers. You will not use or display any trademark, trade name, insignia, or symbols of Columbia, its faculties or departments, or any variation or combination thereof, or the name of any trustee, faculty member, other employee, or student of Columbia, for any purpose whatsoever without Columbia's prior written consent.

7. NO WARRANTY. TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, COLUMBIA DISCLAIMS ALL WARRANTIES AND CONDITIONS, EITHER EXPRESS OR IMPLIED, WITH RESPECT TO THE PROGRAM, INCLUDING BUT NOT LIMITED TO ALL IMPLIED WARRANTIES AND CONDITIONS OF MERCHANTABILITY, TITLE, FITNESS, ADEQUACY OR SUITABILITY FOR A PARTICULAR PURPOSE, USE OR RESULT, OR ARISING FROM A COURSE OF DEALING, USAGE OR TRADE PRACTICE, AND ANY WARRANTIES OF FREEDOM FROM INFRINGEMENT OF ANY DOMESTIC OR FOREIGN PATENTS, COPYRIGHTS, TRADE SECRETS OR OTHER PROPRIETARY RIGHTS OF ANY PARTY. COLUMBIA SPECIFICALLY DISCLAIMS ANY WARRANTY THAT THE FUNCTIONS CONTAINED IN THE PROGRAM WILL MEET YOUR REQUIREMENTS OR WILL OPERATE IN COMBINATIONS OR IN A MANNER SELECTED FOR USE BY YOU, OR THAT THE OPERATION OF THE LICENSED SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE.

8. LIMITATION OF LIABILITY. IN NO EVENT SHALL COLUMBIA BE LIABLE TO YOU FOR ANY DAMAGES RESULTING FROM LOSS OF DATA, LOST PROFITS, LOSS OF USE OF EQUIPMENT OR LOST CONTRACTS OR FOR ANY SPECIAL, INDIRECT, INCIDENTAL, PUNITIVE, EXEMPLARY OR CONSEQUENTIAL DAMAGES IN ANY WAY ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE PROGRAM OR RELATING TO THIS AGREEMENT, HOWEVER CAUSED, EVEN IF COLUMBIA HAS BEEN MADE AWARE OF THE POSSIBILITY OF SUCH DAMAGES. COLUMBIA'S ENTIRE LIABILITY TO YOU, REGARDLESS OF THE FORM OF ANY CLAIM OR ACTION OR THEORY OF LIABILITY (INCLUDING CONTRACT, TORT, OR WARRANTY), SHALL NOT EXCEED IN THE AGGREGATE THE SUM OF TEN U.S. DOLLARS ($10.00).

9. Exports. Each party agrees not to take any action, directly or indirectly, that would violate or cause the other party to violate United States laws and regulations, including, without limitation, regulations and rules regarding sponsored research, trade and import and export controls (the “Export Laws”). In that connection, You confirm to be each of the following:
(a) not a Restricted Party and that no agency of the U.S. Government has denied, suspended, or otherwise abridged the Company’s export or import privileges. A “Restricted Party” means any company or individual on the Department of Treasury Office of Foreign Assets Control list of Specially Designated Nationals and Blocked Persons or List of Foreign Sanctions Evaders, on the Denied Persons List, the Entity List, or the Unverified List maintained by the U.S. Department of Commerce’s Bureau of Industry and Security or on any other list maintained by any governmental agency restricting the export of any items to or other transactions with specific individuals, companies or other entities;
(b) not directly or indirectly owned or controlled by or acting on behalf of others whose interests taken in the aggregate make them subject to U.S. trade sanctions or restrictions;
(c) not directly or indirectly owned or controlled by or acting on behalf of a government of or entity located in a country subject to economic sanctions programs that are or may be maintained by the U.S. Government; and
(d) not otherwise restricted, embargoed, or prohibited under applicable law from entering into agreements with U.S. entities and individuals.
You shall not export, re-export or otherwise transfer to any individuals or entities identified in items (i)-(iv) above any hardware, software, technology or services provided by Columbia under this Agreement. You confirm that you do not intend for the hardware, software, technology or services that Columbia provides under this Agreement to be used for any purposes prohibited by U.S. export control laws and regulations, including without limitation nuclear, chemical, or biological weapons proliferation, or for military end-uses or military end-users. The provisions of this section will remain in full force and effect during the term of this Agreement, and the You will immediately notify Columbia of any events or changes that may conflict with the assurances and statements provided hereunder.
You agree to comply with all applicable export laws and regulations of all jurisdictions with respect to the Program and obtain, at your own expense, any required permits or export clearances, copies of which you shall provide to Columbia prior to such export.

10. U.S. Government Agencies. If You are an agency of the United States Government, the Program constitutes "commercial computer software" or "commercial computer software documentation." Absent a written agreement to the contrary, the Government's rights with respect to the Program are limited by the terms of this Agreement, pursuant to FAR 12.212(a) and/or DFARS 227.7202-4, as applicable.

11. Assignment. Neither this Agreement nor any rights, obligations, or licenses granted hereunder may be assigned or delegated by You without the prior written consent of Columbia. This Agreement shall inure to the benefit of the parties and their permitted successors and assigns.

12. Governing Law; Jurisdiction and Venue. This Agreement shall be governed by New York law applicable to agreements made and to be fully performed in New York, without reference to the conflict of laws principles of any jurisdiction. The parties agree that any and all claims arising under this Agreement or relating thereto shall be heard and determined either in the United States District Court for the Southern District of New York or in the Courts of the State of New York located in the City and County of New York, and the parties agree to submit themselves to the personal jurisdiction of those Courts and to waive any objections as to the convenience of the forum.

13. Severability. If any provision of this Agreement shall be held by a court of competent jurisdiction to be illegal, invalid or unenforceable, the remaining provisions shall remain in full force and effect.

14. Miscellaneous. (a) This Agreement and its exhibits contain the entire understanding and agreement between the parties respecting the subject matter hereof. (b) This Agreement may not be supplemented, modified, amended, released or discharged except by an instrument in writing signed by each party’s duly authorized representative. (c) All captions and headings in this Agreement are for purposes of convenience only and shall not affect the construction or interpretation of any of its provisions. (d) Any waiver by either party of any default or breach hereunder shall not constitute a waiver of any provision of this Agreement or of any subsequent default or breach of the same or a different kind. (e) This Agreement shall be binding upon and shall inure to the benefit of the parties, their successors, and permitted assigns.