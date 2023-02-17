// SPDX-License-Identifier: Apache-2.0

package firrtl.passes

import firrtl.FirrtlUserException

// Error handling
class PassException(message: String) extends FirrtlUserException(message)
